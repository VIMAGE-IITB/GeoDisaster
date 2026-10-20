from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_PASS_THROUGH_TOOLS = {"GoogleSearch", "Plot", "Solver", "Calculator"}
_LOCK = threading.Lock()
_CACHE: dict[tuple[str, int], dict[int, "Scenario"]] = {}
_SESSION_STATE: dict[str, "SessionState"] = {}


@dataclass
class ScenarioTurn:
    tool: str
    args: dict[str, Any]
    response: dict[str, Any]
    semantic_key: str = ""


@dataclass
class Scenario:
    case_id: int
    idx: int
    question: str
    turns: list[ScenarioTurn]
    tool_index: dict[str, list[ScenarioTurn]] = field(default_factory=dict)
    layer_semantics: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionState:
    cursor: int = 0
    successful_keys: set[str] = field(default_factory=set)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_counters: dict[str, int] = field(default_factory=lambda: {"gpkg": 0, "img": 0, "tif": 0})
    artifact_aliases: dict[str, str] = field(default_factory=dict)
    layers: dict[str, set[str]] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    layer_semantics: dict[str, str] = field(default_factory=dict)


def is_enabled() -> bool:
    return str(os.environ.get("MAS_SCRIPTED_TOOL_ENABLE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _pass_through_tools() -> set[str]:
    raw = os.environ.get("MAS_SCRIPTED_TOOL_PASS_THROUGH_TOOLS")
    if raw is None:
        return set(_DEFAULT_PASS_THROUGH_TOOLS)
    return {part.strip() for part in raw.split(",") if part.strip()}


def maybe_handle(tool_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if not is_enabled() or tool_name in _pass_through_tools():
        return None

    session_id = str(params.get("__scenario_session_id", "")).strip()
    case_id_raw = params.get("__scenario_case_id")
    if not session_id or case_id_raw is None:
        return None

    try:
        case_id = int(case_id_raw)
    except Exception:
        return None

    original_args = params.get("__scenario_original_args")
    if not isinstance(original_args, dict):
        original_args = {k: v for k, v in params.items() if not str(k).startswith("__scenario_")}

    scenarios = _load_scenarios(
        Path(str(os.environ.get("MAS_SCRIPTED_TOOL_DATA_PATH", "")).strip()),
        int(os.environ.get("MAS_SCRIPTED_TOOL_CASE_LIMIT", "5") or 5),
    )
    scenario = scenarios.get(case_id)
    if scenario is None:
        return None

    with _LOCK:
        state = _SESSION_STATE.setdefault(session_id, SessionState())
        if "__scenario_layer_semantics__" not in state.artifacts:
            state.artifacts["__scenario_layer_semantics__"] = dict(scenario.layer_semantics)
        matched = _match_turn(tool_name=tool_name, args=original_args, scenario=scenario, state=state)
        if matched is None:
            return {
                "text": f"Error in toolcall: no scripted semantic match for {tool_name}.",
                "error_code": 1,
            }

        response = _materialize_response(
            tool_name=tool_name,
            args=original_args,
            response=matched.response,
            state=state,
        )
        state.successful_keys.add(matched.semantic_key or _semantic_key(tool_name, original_args))
        state.cursor = max(state.cursor, _turn_position(scenario, matched) + 1)
        _absorb_response_state(args=original_args, response=response, state=state)
        return response


def _load_scenarios(data_path: Path, case_limit: int) -> dict[int, Scenario]:
    key = (str(data_path.resolve()), case_limit)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if not data_path.exists():
        raise FileNotFoundError(f"Scripted tool data file not found: {data_path}")

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        cases = payload["cases"]
        scenarios = {int(case["case_id"]): _scenario_from_compiled_case(case) for case in cases[:case_limit or None]}
        _CACHE[key] = scenarios
        return scenarios

    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Unsupported scripted tool data schema in {data_path}")

    scenarios: dict[int, Scenario] = {}
    selected_records = records[:case_limit] if case_limit else records
    for case_id, record in enumerate(selected_records):
        if isinstance(record, dict) and isinstance(record.get("executions"), list):
            scenarios[case_id] = _scenario_from_execution_record(case_id, record, data_path.parent)
        else:
            scenarios[case_id] = _scenario_from_record(case_id, record, data_path.parent)
    _CACHE[key] = scenarios
    return scenarios


def _scenario_from_compiled_case(case: dict[str, Any]) -> Scenario:
    turns = [
        ScenarioTurn(
            tool=str(turn.get("tool", "")).strip(),
            args=dict(turn.get("args") or {}),
            response=dict(turn.get("response") or {}),
            semantic_key=str(turn.get("semantic_key") or _semantic_key(str(turn.get("tool", "")).strip(), dict(turn.get("args") or {}))),
        )
        for turn in (case.get("turns") or [])
        if str(turn.get("tool", "")).strip()
    ]
    tool_index: dict[str, list[ScenarioTurn]] = {}
    layer_semantics = _collect_scenario_layer_semantics(turns)
    for turn in turns:
        tool_index.setdefault(turn.tool, []).append(turn)
    return Scenario(
        case_id=int(case.get("case_id", 0)),
        idx=int(case.get("idx", case.get("case_id", 0))),
        question=str(case.get("question", "")).strip(),
        turns=turns,
        tool_index=tool_index,
        layer_semantics=layer_semantics,
    )


def _scenario_from_record(case_id: int, record: dict[str, Any], data_dir: Path) -> Scenario:
    conversation = list(record.get("conversation") or [])
    turns: list[ScenarioTurn] = []
    for idx, item in enumerate(conversation):
        if str(item.get("from", "")).lower() not in {"gpt", "assistant"}:
            continue
        payload = _extract_first_json_object(str(item.get("value", "")))
        if not isinstance(payload, dict):
            continue
        actions = payload.get("actions") or []
        if not isinstance(actions, list) or not actions:
            continue
        action = actions[0]
        if not isinstance(action, dict):
            continue
        tool_name = str(action.get("name", "")).strip()
        if not tool_name or tool_name == "Terminate":
            continue
        next_obs = _find_next_observation(conversation[idx + 1 :])
        if not next_obs:
            continue
        args = dict(action.get("arguments") or {})
        response = _response_from_observation(
            case_id=case_id,
            data_dir=data_dir,
            tool_name=tool_name,
            args=args,
            observation_text=next_obs,
        )
        turns.append(ScenarioTurn(tool=tool_name, args=args, response=response, semantic_key=_semantic_key(tool_name, args)))
    tool_index: dict[str, list[ScenarioTurn]] = {}
    layer_semantics = _collect_scenario_layer_semantics(turns)
    for turn in turns:
        tool_index.setdefault(turn.tool, []).append(turn)
    return Scenario(case_id=case_id, idx=int(record.get("idx", case_id)), question=str(record.get("question", "")).strip(), turns=turns, tool_index=tool_index, layer_semantics=layer_semantics)


def _scenario_from_execution_record(case_id: int, record: dict[str, Any], data_dir: Path) -> Scenario:
    turns: list[ScenarioTurn] = []
    for execution in record.get("executions") or []:
        for item in execution.get("turns") or []:
            action = dict(item.get("action") or {})
            tool_name = str(action.get("tool", "")).strip()
            if not tool_name or tool_name == "TaskComplete":
                continue
            args = dict(action.get("args") or {})
            response = _response_from_explicit_output(
                case_id=case_id,
                data_dir=data_dir,
                tool_name=tool_name,
                args=args,
                tool_output=dict(item.get("tool_output") or {}),
            )
            turns.append(ScenarioTurn(tool=tool_name, args=args, response=response, semantic_key=_semantic_key(tool_name, args)))
    tool_index: dict[str, list[ScenarioTurn]] = {}
    layer_semantics = _collect_scenario_layer_semantics(turns)
    for turn in turns:
        tool_index.setdefault(turn.tool, []).append(turn)
    return Scenario(case_id=case_id, idx=int(record.get("idx", case_id)), question=str(record.get("question", "")).strip(), turns=turns, tool_index=tool_index, layer_semantics=layer_semantics)


def _match_turn(*, tool_name: str, args: dict[str, Any], scenario: Scenario, state: SessionState) -> ScenarioTurn | None:
    candidates = list(scenario.tool_index.get(tool_name, []))
    if not candidates:
        return None

    semantic = _semantic_key(tool_name, args)
    exact_semantic = [turn for turn in candidates if turn.semantic_key == semantic]
    ordered = exact_semantic + [turn for turn in candidates if turn.semantic_key != semantic]
    for turn in ordered:
        if _semantic_match(tool_name=tool_name, expected=turn.args, actual=args, state=state):
            return turn
    return None


def _semantic_match(*, tool_name: str, expected: dict[str, Any], actual: dict[str, Any], state: SessionState) -> bool:
    if tool_name == "GetAreaBoundary":
        return _normalize_text(actual.get("area")) == _normalize_text(expected.get("area"))

    if tool_name == "AddPoisLayer":
        return (
            _gpkg_available(actual.get("gpkg"), state)
            and _normalize_gpkg(actual.get("gpkg")) == _normalize_gpkg(expected.get("gpkg"))
            and _normalize_single_tag(actual.get("query")) == _normalize_single_tag(expected.get("query"))
            and bool(str(actual.get("layer_name", "")).strip())
        )

    if tool_name == "AddIndexLayer":
        return (
            _gpkg_available(actual.get("gpkg"), state)
            and
            _normalize_gpkg(actual.get("gpkg")) == _normalize_gpkg(expected.get("gpkg"))
            and _normalize_text(actual.get("index_type")) == _normalize_text(expected.get("index_type"))
            and _safe_int(actual.get("year")) == _safe_int(expected.get("year"))
            and _safe_int(actual.get("month")) == _safe_int(expected.get("month"))
            and bool(str(actual.get("layer_name", "")).strip())
        )

    if tool_name == "ComputeIndexChange":
        scenario_semantics = state.artifacts.get("__scenario_layer_semantics__", {})
        expected_l1 = _resolve_layer_semantic(expected.get("layer1_name"), state, scenario_layer_semantics=scenario_semantics, fallback_from_name=True)
        expected_l2 = _resolve_layer_semantic(expected.get("layer2_name"), state, scenario_layer_semantics=scenario_semantics, fallback_from_name=True)
        actual_l1 = _resolve_layer_semantic(actual.get("layer1_name"), state)
        actual_l2 = _resolve_layer_semantic(actual.get("layer2_name"), state)
        return (
            _gpkg_available(actual.get("gpkg"), state)
            and
            _normalize_gpkg(actual.get("gpkg")) == _normalize_gpkg(expected.get("gpkg"))
            and _normalize_text(actual.get("index_type")) == _normalize_text(expected.get("index_type"))
            and expected_l1 == actual_l1
            and expected_l2 == actual_l2
            and bool(str(actual.get("diff_layer_name", "")).strip())
        )

    if tool_name == "ComputeDistance":
        expected_src = _resolve_layer_semantic(expected.get("src_layer"), state, scenario_layer_semantics=state.artifacts.get("__scenario_layer_semantics__", {}), fallback_from_name=True)
        expected_tar = _resolve_layer_semantic(expected.get("tar_layer"), state, scenario_layer_semantics=state.artifacts.get("__scenario_layer_semantics__", {}), fallback_from_name=True)
        actual_src = _resolve_layer_semantic(actual.get("src_layer"), state)
        actual_tar = _resolve_layer_semantic(actual.get("tar_layer"), state)
        return (
            _gpkg_available(actual.get("gpkg"), state)
            and _normalize_gpkg(actual.get("gpkg")) == _normalize_gpkg(expected.get("gpkg"))
            and expected_src == actual_src
            and expected_tar == actual_tar
        )

    if tool_name in {"DisplayOnMap", "DisplayOnGeotiff"}:
        if not _gpkg_available(actual.get("gpkg"), state):
            return False
        if tool_name == "DisplayOnGeotiff" and not _same_artifact_ref(actual.get("geotiff"), expected.get("geotiff"), state):
            return False
        return _layers_match_expected(expected.get("layers"), actual.get("layers"), state)

    if tool_name == "GetBboxFromGeotiff":
        return _same_artifact_ref(actual.get("geotiff"), expected.get("geotiff"), state)

    if tool_name == "ShowIndexLayer":
        scenario_semantics = state.artifacts.get("__scenario_layer_semantics__", {})
        expected_name = str(expected.get("layer_name", "")).strip()
        actual_name = str(actual.get("layer_name", "")).strip()
        expected_layer = _resolve_layer_semantic(expected.get("layer_name"), state, scenario_layer_semantics=scenario_semantics, fallback_from_name=True)
        actual_layer = _resolve_layer_semantic(actual.get("layer_name"), state)
        layer_exists = _layer_exists(actual.get("layer_name"), state)
        if not layer_exists and expected_name in scenario_semantics:
            return False
        if not layer_exists:
            return (
                _gpkg_available(actual.get("gpkg"), state)
                and _normalize_gpkg(actual.get("gpkg")) == _normalize_gpkg(expected.get("gpkg"))
                and _normalize_text(actual.get("index_type")) == _normalize_text(expected.get("index_type"))
                and _normalize_text(actual_name) == _normalize_text(expected_name)
            )
        return (
            _gpkg_available(actual.get("gpkg"), state)
            and _normalize_gpkg(actual.get("gpkg")) == _normalize_gpkg(expected.get("gpkg"))
            and _normalize_text(actual.get("index_type")) == _normalize_text(expected.get("index_type"))
            and expected_layer == actual_layer
        )

    if tool_name in {"TextToBbox", "SegmentObjectPixels", "CountGivenObject"}:
        return _same_artifact_ref(actual.get("image"), expected.get("image"), state) and _normalize_text(actual.get("text")) == _normalize_text(expected.get("text"))

    if tool_name == "RegionAttributeDescription":
        return _same_artifact_ref(actual.get("image"), expected.get("image"), state) and _normalize_text(actual.get("attribute")) == _normalize_text(expected.get("attribute"))

    if tool_name == "ChangeDetection":
        return (
            _same_artifact_ref(actual.get("pre_image"), expected.get("pre_image"), state)
            and _same_artifact_ref(actual.get("post_image"), expected.get("post_image"), state)
            and _normalize_text(actual.get("text")) == _normalize_text(expected.get("text"))
        )

    if tool_name == "DrawBox":
        if not _same_artifact_ref(actual.get("image"), expected.get("image"), state):
            return False
        expected_bbox = _normalize_bbox(expected.get("bbox"))
        return bool(actual.get("annotation")) and (not expected_bbox or _normalize_bbox(actual.get("bbox")) == expected_bbox)

    if tool_name == "AddText":
        if not _same_artifact_ref(actual.get("image"), expected.get("image"), state):
            return False
        if not str(actual.get("position", "")).strip() or not str(actual.get("text", "")).strip():
            return False
        return _text_preserves_numbers(expected.get("text"), actual.get("text"))

    if tool_name in {"ObjectDetection", "OCR", "ImageDescription"}:
        return _same_artifact_ref(actual.get("image"), expected.get("image"), state)

    return _deep_equal(expected, actual)


def _materialize_response(*, tool_name: str, args: dict[str, Any], response: dict[str, Any], state: SessionState) -> dict[str, Any]:
    materialized = json.loads(json.dumps(response))
    produced = materialized.get("produced")
    if not isinstance(produced, dict):
        produced = {"artifacts": [], "layers": [], "fields": {}}
        materialized["produced"] = produced

    if tool_name == "AddPoisLayer":
        layer_name = str(args.get("layer_name", "")).strip() or "poi_layer"
        query = dict(args.get("query") or {})
        count_value = _extract_count_from_response(materialized)
        produced["layers"] = [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": layer_name}]
        produced["fields"] = {f"{layer_name}_count": count_value} if count_value is not None else {}
        materialized["text"] = _replace_layer_name_in_text(str(materialized.get("text", "")), layer_name)
        state.layer_semantics[layer_name] = _semantic_key(tool_name, args)
        return materialized

    if tool_name == "AddIndexLayer":
        layer_name = str(args.get("layer_name", "")).strip() or _default_index_layer_name(args)
        produced["layers"] = [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": layer_name}]
        state.layer_semantics[layer_name] = _semantic_key(tool_name, args)
        return materialized

    if tool_name == "ComputeIndexChange":
        diff_name = str(args.get("diff_layer_name", "")).strip() or "index_change"
        semantic = _index_change_semantic_from_state(args, state)
        if produced.get("layers"):
            produced["layers"] = [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": diff_name}]
        else:
            produced["layers"] = [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": diff_name}]
        state.layer_semantics[diff_name] = semantic
        return materialized

    if tool_name == "ComputeDistance":
        src = str(args.get("src_layer", "")).strip()
        tar = str(args.get("tar_layer", "")).strip()
        distance_layer = f"{src}_to_{tar}_distances"
        reverse_distance_layer = f"{tar}_to_{src}_distances"
        semantic = _distance_semantic_from_state(args, state)
        produced["layers"] = [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": distance_layer}] if src and tar else produced.get("layers", [])
        if src and tar:
            state.layer_semantics[distance_layer] = semantic
            state.layer_semantics[reverse_distance_layer] = semantic
        return materialized

    if tool_name == "GetBboxFromGeotiff":
        produced.setdefault("fields", {})
        produced["fields"].setdefault("bbox", "mock_bbox")
        if not materialized.get("text"):
            materialized["text"] = "Returned bbox for provided GeoTIFF."
        return materialized

    if tool_name in {"DisplayOnMap", "DisplayOnGeotiff", "ShowIndexLayer", "DrawBox", "AddText", "Plot"}:
        artifacts = list(produced.get("artifacts") or [])
        if not artifacts:
            artifacts.append({"kind": "img", "path": _image_path_from_response(materialized, tool_name), "note": f"{tool_name} output"})
        produced["artifacts"] = artifacts
        return materialized

    return materialized


def _absorb_response_state(*, args: dict[str, Any], response: dict[str, Any], state: SessionState) -> None:
    produced = response.get("produced") or {}
    for art in produced.get("artifacts", []) or []:
        handle = str(art.get("handle") or art.get("gpkg") or "").strip()
        if not handle:
            kind = str(art.get("kind", "img")).strip() or "img"
            state.artifact_counters[kind] = state.artifact_counters.get(kind, 0) + 1
            prefix = "image" if kind == "img" else kind
            handle = f"{prefix}_{state.artifact_counters[kind]}"
        if handle:
            state.artifacts[handle] = dict(art)
            path = str(art.get("path", "")).strip()
            if path:
                state.artifact_aliases[_normalize_artifact_ref(path)] = handle
                state.artifact_aliases[_normalize_artifact_ref(Path(path).name)] = handle
    for layer in produced.get("layers", []) or []:
        gpkg = str(layer.get("gpkg", "")).strip()
        name = str(layer.get("name", "")).strip()
        if gpkg and name:
            state.layers.setdefault(gpkg, set()).add(name)
    for key, value in (produced.get("fields") or {}).items():
        state.fields[str(key)] = value

    if "gpkg" in response:
        state.artifact_counters["gpkg"] = max(state.artifact_counters.get("gpkg", 0), 1)
        state.artifacts.setdefault("gpkg_1", {"kind": "gpkg", "path": response["gpkg"]})


def _turn_position(scenario: Scenario, turn: ScenarioTurn) -> int:
    try:
        return scenario.turns.index(turn)
    except ValueError:
        return 0


def _semantic_key(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "GetAreaBoundary":
        return f"GetAreaBoundary|area={_normalize_text(args.get('area'))}"
    if tool_name == "AddPoisLayer":
        return f"AddPoisLayer|gpkg={_normalize_gpkg(args.get('gpkg'))}|query={_normalize_single_tag(args.get('query'))}"
    if tool_name == "AddIndexLayer":
        return (
            f"AddIndexLayer|gpkg={_normalize_gpkg(args.get('gpkg'))}|index={_normalize_text(args.get('index_type'))}"
            f"|year={_safe_int(args.get('year'))}|month={_safe_int(args.get('month'))}"
        )
    if tool_name == "ComputeIndexChange":
        return (
            f"ComputeIndexChange|gpkg={_normalize_gpkg(args.get('gpkg'))}|index={_normalize_text(args.get('index_type'))}"
            f"|l1={_normalize_text(args.get('layer1_name'))}|l2={_normalize_text(args.get('layer2_name'))}"
        )
    if tool_name == "ComputeDistance":
        return f"ComputeDistance|gpkg={_normalize_gpkg(args.get('gpkg'))}|src={_normalize_text(args.get('src_layer'))}|tar={_normalize_text(args.get('tar_layer'))}"
    if tool_name == "GetBboxFromGeotiff":
        return f"GetBboxFromGeotiff|geotiff={_normalize_text(args.get('geotiff'))}"
    if tool_name in {"TextToBbox", "SegmentObjectPixels", "CountGivenObject"}:
        return f"{tool_name}|image={_normalize_text(args.get('image'))}|text={_normalize_text(args.get('text'))}"
    if tool_name == "RegionAttributeDescription":
        return f"RegionAttributeDescription|image={_normalize_text(args.get('image'))}|attribute={_normalize_text(args.get('attribute'))}"
    if tool_name == "ChangeDetection":
        return f"ChangeDetection|pre={_normalize_text(args.get('pre_image'))}|post={_normalize_text(args.get('post_image'))}|text={_normalize_text(args.get('text'))}"
    return f"{tool_name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def _resolve_layer_semantic(
    layer_name: Any,
    state: SessionState,
    scenario_layer_semantics: dict[str, str] | None = None,
    fallback_from_name: bool = False,
) -> str:
    name = str(layer_name or "").strip()
    if name in state.layer_semantics:
        return state.layer_semantics[name]
    if scenario_layer_semantics and name in scenario_layer_semantics:
        return scenario_layer_semantics[name]
    if fallback_from_name:
        year = _year_from_layer_name(name)
        month = _month_from_layer_name(name)
        index = _index_type_from_layer_name(name)
        if index:
            return _semantic_key("AddIndexLayer", {"gpkg": "gpkg_1", "index_type": index, "year": year, "month": month, "layer_name": name})
    return _normalize_text(name)


def _collect_scenario_layer_semantics(turns: list[ScenarioTurn]) -> dict[str, str]:
    out: dict[str, str] = {}
    for turn in turns:
        if turn.tool == "AddPoisLayer":
            layer_name = str(turn.args.get("layer_name", "")).strip()
            if layer_name:
                out[layer_name] = _semantic_key(turn.tool, turn.args)
        elif turn.tool == "AddIndexLayer":
            layer_name = str(turn.args.get("layer_name", "")).strip()
            if layer_name:
                out[layer_name] = _semantic_key(turn.tool, turn.args)
        elif turn.tool == "ComputeIndexChange":
            diff_name = str(turn.args.get("diff_layer_name", "")).strip()
            if diff_name:
                out[diff_name] = _index_change_semantic_from_mapping(turn.args, out)
        elif turn.tool == "ComputeDistance":
            src = str(turn.args.get("src_layer", "")).strip()
            tar = str(turn.args.get("tar_layer", "")).strip()
            if src and tar:
                semantic = _distance_semantic_from_mapping(turn.args, out)
                out[f"{src}_to_{tar}_distances"] = semantic
                out[f"{tar}_to_{src}_distances"] = semantic
    return out


def _layer_semantic_from_mapping(name: Any, mapping: dict[str, str]) -> str:
    layer_name = str(name or "").strip()
    if layer_name in mapping:
        return mapping[layer_name]
    year = _year_from_layer_name(layer_name)
    month = _month_from_layer_name(layer_name)
    index = _index_type_from_layer_name(layer_name)
    if index:
        return _semantic_key("AddIndexLayer", {"gpkg": "gpkg_1", "index_type": index, "year": year, "month": month, "layer_name": layer_name})
    return _normalize_text(layer_name)


def _distance_semantic_from_mapping(args: dict[str, Any], mapping: dict[str, str]) -> str:
    src = _layer_semantic_from_mapping(args.get("src_layer"), mapping)
    tar = _layer_semantic_from_mapping(args.get("tar_layer"), mapping)
    return f"ComputeDistance|gpkg={_normalize_gpkg(args.get('gpkg'))}|src={src}|tar={tar}"


def _distance_semantic_from_state(args: dict[str, Any], state: SessionState) -> str:
    src = _resolve_layer_semantic(args.get("src_layer"), state)
    tar = _resolve_layer_semantic(args.get("tar_layer"), state)
    return f"ComputeDistance|gpkg={_normalize_gpkg(args.get('gpkg'))}|src={src}|tar={tar}"


def _index_change_semantic_from_mapping(args: dict[str, Any], mapping: dict[str, str]) -> str:
    l1 = _layer_semantic_from_mapping(args.get("layer1_name"), mapping)
    l2 = _layer_semantic_from_mapping(args.get("layer2_name"), mapping)
    return f"ComputeIndexChange|gpkg={_normalize_gpkg(args.get('gpkg'))}|index={_normalize_text(args.get('index_type'))}|l1={l1}|l2={l2}"


def _index_change_semantic_from_state(args: dict[str, Any], state: SessionState) -> str:
    l1 = _resolve_layer_semantic(args.get("layer1_name"), state)
    l2 = _resolve_layer_semantic(args.get("layer2_name"), state)
    return f"ComputeIndexChange|gpkg={_normalize_gpkg(args.get('gpkg'))}|index={_normalize_text(args.get('index_type'))}|l1={l1}|l2={l2}"


def _gpkg_available(value: Any, state: SessionState) -> bool:
    gpkg = _normalize_gpkg(value)
    return gpkg in state.artifacts


def _layers_match_expected(expected_layers: Any, actual_layers: Any, state: SessionState) -> bool:
    if not isinstance(expected_layers, list) or not isinstance(actual_layers, list) or not actual_layers:
        return False
    scenario_semantics = state.artifacts.get("__scenario_layer_semantics__", {})
    expected_semantics = {
        _resolve_layer_semantic(layer, state, scenario_layer_semantics=scenario_semantics, fallback_from_name=True)
        for layer in expected_layers
    }
    if any(not _layer_exists(layer, state) for layer in actual_layers):
        return False
    actual_semantics = {
        _resolve_layer_semantic(layer, state, fallback_from_name=True)
        for layer in actual_layers
    }
    return expected_semantics.issubset(actual_semantics)


def _layer_exists(layer_name: Any, state: SessionState) -> bool:
    name = str(layer_name or "").strip()
    if not name:
        return False
    if name in state.layer_semantics:
        return True
    return any(name in names for names in state.layers.values())


def _same_artifact_ref(actual: Any, expected: Any, state: SessionState | None = None) -> bool:
    return _normalize_artifact_ref(actual, state) == _normalize_artifact_ref(expected, state)


def _normalize_artifact_ref(value: Any, state: SessionState | None = None) -> str:
    text = _normalize_text(value).replace("\\n", "").rstrip(".")
    match = re.fullmatch(r"img_(\d+)", text)
    if match:
        return f"image_{match.group(1)}"
    match = re.fullmatch(r"geotiff_(\d+)", text)
    if match:
        return f"tif_{match.group(1)}"
    match = re.fullmatch(r"tiff_(\d+)", text)
    if match:
        return f"tif_{match.group(1)}"
    if state is not None and text in state.artifact_aliases:
        return state.artifact_aliases[text]
    return text


def _text_preserves_numbers(expected: Any, actual: Any) -> bool:
    expected_nums = re.findall(r"[-+]?\d*\.?\d+", str(expected or ""))
    if not expected_nums:
        return True
    actual_text = str(actual or "")
    return all(num in actual_text for num in expected_nums)


def _image_path_from_response(response: dict[str, Any], tool_name: str) -> str:
    text = str(response.get("text", ""))
    for match in re.findall(r"(?:saved to|->)\s+([^\s,;]+)", text, flags=re.I):
        path = match.replace("\\n", "").strip().strip("\"'").rstrip(".")
        if path:
            return path
    stripped = text.replace("\\n", "").strip().strip("\"'").rstrip(".")
    if re.search(r"\.(png|jpg|jpeg|tif|tiff)$", stripped, flags=re.I):
        return stripped
    return f"MOCK_IMG_{tool_name}.png"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _normalize_gpkg(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "gpkg_1"
    digit_match = re.fullmatch(r"gpkg_(\d+)", text)
    if digit_match:
        return f"gpkg_{digit_match.group(1)}"
    path_match = re.search(r"gpkg[_-]?(\d+)", text, flags=re.I)
    if path_match:
        return f"gpkg_{path_match.group(1)}"
    if "gpkg" in text.lower() or text.lower().endswith(".gpkg"):
        return "gpkg_1"
    return text


def _normalize_single_tag(query: Any) -> str:
    if not isinstance(query, dict) or len(query) != 1:
        return ""
    key, value = next(iter(query.items()))
    return f"{_normalize_text(key)}={_normalize_text(value)}"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _extract_count_from_response(response: dict[str, Any]) -> int | None:
    produced = response.get("produced") or {}
    for value in (produced.get("fields") or {}).values():
        if isinstance(value, int):
            return value
    return _first_int(str(response.get("text", "")))


def _replace_layer_name_in_text(text: str, layer_name: str) -> str:
    if "'" in text:
        return re.sub(r"'[^']+'", f"'{layer_name}'", text, count=1)
    return text


def _default_index_layer_name(args: dict[str, Any]) -> str:
    index_type = str(args.get("index_type", "index")).lower()
    year = _safe_int(args.get("year"))
    month = _safe_int(args.get("month"))
    if year and month:
        return f"{index_type}_{year}_{month:02d}"
    if year:
        return f"{index_type}_{year}"
    return index_type


def _year_from_layer_name(name: Any) -> int | None:
    match = re.search(r"(20\d{2})", str(name or ""))
    return int(match.group(1)) if match else None


def _month_from_layer_name(name: Any) -> int | None:
    text = str(name or "")
    match = re.search(r"(?:_|-)(0?[1-9]|1[0-2])(?:$|[^0-9])", text)
    return int(match.group(1)) if match else None


def _index_type_from_layer_name(name: Any) -> str:
    text = _normalize_text(name)
    for token in ("ndvi", "ndbi", "nbr", "ndwi", "evi"):
        if token in text:
            return token.upper()
    return ""


def _find_next_observation(conversation_tail: list[dict[str, Any]]) -> str:
    for item in conversation_tail:
        if str(item.get("from", "")).lower() not in {"human", "user"}:
            continue
        value = str(item.get("value", ""))
        if "OBSERVATION:" in value:
            text = value.split("OBSERVATION:", 1)[1]
            text = text.split("Please summarize the model outputs and answer my first question.", 1)[0]
            return text.strip()
    return ""


def _response_from_observation(
    *,
    case_id: int,
    data_dir: Path,
    tool_name: str,
    args: dict[str, Any],
    observation_text: str,
) -> dict[str, Any]:
    response: dict[str, Any] = {"text": observation_text, "error_code": 0}

    if tool_name == "GetAreaBoundary":
        gpkg_path = _ensure_placeholder_gpkg(data_dir, case_id)
        response["gpkg"] = gpkg_path
        response["produced"] = {
            "artifacts": [{"kind": "gpkg", "path": gpkg_path, "note": "scenario boundary"}],
            "layers": [{"gpkg": "__autogen__", "name": "area_boundary"}],
            "fields": {},
        }
        return response

    if tool_name == "AddPoisLayer":
        layer_name = str(args.get("layer_name", "")).strip()
        count = _first_int(observation_text)
        response["produced"] = {
            "artifacts": [],
            "layers": [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": layer_name}] if layer_name else [],
            "fields": {f"{layer_name}_count": count} if layer_name and count is not None else {},
        }
        return response

    if tool_name == "ComputeDistance":
        src = str(args.get("src_layer", "")).strip()
        tar = str(args.get("tar_layer", "")).strip()
        pair, distance_m, travel_time_s = _parse_first_distance(observation_text)
        fields: dict[str, Any] = {}
        if pair:
            fields["pair"] = pair
        if distance_m is not None:
            fields["distance_m"] = distance_m
        if travel_time_s is not None:
            fields["travel_time_s"] = travel_time_s
        response["produced"] = {
            "artifacts": [],
            "layers": [{"gpkg": str(args.get("gpkg", "gpkg_1")), "name": f"{src}_to_{tar}_distances"}] if src and tar else [],
            "fields": fields,
        }
        return response

    return response


def _response_from_explicit_output(
    *,
    case_id: int,
    data_dir: Path,
    tool_name: str,
    args: dict[str, Any],
    tool_output: dict[str, Any],
) -> dict[str, Any]:
    ok = bool(tool_output.get("ok", True))
    response: dict[str, Any] = {
        "text": str(tool_output.get("text", "")),
        "error_code": 0 if ok else 1,
    }
    produced = tool_output.get("produced")
    if isinstance(produced, dict):
        response["produced"] = json.loads(json.dumps(produced))

    if tool_name == "GetAreaBoundary":
        gpkg_path = _ensure_placeholder_gpkg(data_dir, case_id)
        response["gpkg"] = gpkg_path
        if "produced" not in response:
            response["produced"] = {
                "artifacts": [{"kind": "gpkg", "path": gpkg_path, "note": "scenario boundary"}],
                "layers": [{"gpkg": "__autogen__", "name": "area_boundary"}],
                "fields": {},
            }
        else:
            artifacts = list(response["produced"].get("artifacts") or [])
            if not artifacts:
                artifacts.append({"kind": "gpkg", "path": gpkg_path, "note": "scenario boundary"})
            else:
                for artifact in artifacts:
                    if artifact.get("kind") == "gpkg":
                        artifact["path"] = gpkg_path
            response["produced"]["artifacts"] = artifacts
        return response

    return response


def _ensure_placeholder_gpkg(data_dir: Path, case_id: int) -> str:
    out_dir = data_dir / "replay_artifacts" / "scripted_gpks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{case_id}_gpkg_1.gpkg"
    path.touch(exist_ok=True)
    return str(path.resolve())


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : idx + 1])
                except Exception:
                    return None
    return None


def _first_int(text: str) -> int | None:
    match = re.search(r"(\d+)", str(text or ""))
    return int(match.group(1)) if match else None


def _parse_first_distance(text: str) -> tuple[str, float | None, float | None]:
    match = re.search(
        r"\n([^,\n]+?)\s*,\s*([^,\n]+?),\s*distance=([-+]?\d*\.?\d+)\s*m(?:,\s*travel_time=([-+]?\d*\.?\d+)\s*s)?",
        "\n" + str(text or ""),
    )
    if not match:
        return "", None, None
    pair = f"{match.group(1).strip()} -> {match.group(2).strip()}"
    distance_m = round(float(match.group(3)), 2)
    travel_time_s = round(float(match.group(4)), 2) if match.group(4) else None
    return pair, distance_m, travel_time_s


def _normalize_bbox(value: Any) -> str:
    nums = re.findall(r"-?\d+", str(value or ""))
    return ",".join(nums[:4]) if len(nums) >= 4 else ""


def _deep_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        if set(lhs.keys()) != set(rhs.keys()):
            return False
        return all(_deep_equal(lhs[k], rhs[k]) for k in lhs)
    if isinstance(lhs, list) and isinstance(rhs, list):
        return len(lhs) == len(rhs) and all(_deep_equal(a, b) for a, b in zip(lhs, rhs))
    return lhs == rhs
