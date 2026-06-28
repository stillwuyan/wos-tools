from log import get_logger
import time
from pathlib import Path

import importlib

from utils import _apply_jitter

logger = get_logger(__name__)


def _execute_action(ui, steps, context_key, src_dir: Path, cfg, depth: int = 0) -> bool:
    """Execute a list of step tuples for the given context key.

    Returns True if execution completed and may be repeated; returns False
    if a 'locate' action failed (meaning stop processing this entry and move on).
    """
    loc_coord = None
    total_actions = len(steps) if steps is not None else 0
    start_time = time.perf_counter()
    ok_count = skip_count = fail_count = 0
    result = True

    # Build log prefixes for tree-style nested output
    if depth == 0:
        sp = '  '          # step prefix
        start_pfx = ''     # TaskStart prefix
        end_pfx = ''       # TaskEnd prefix
    else:
        base = '  ' * depth
        sp = base + '│  '
        start_pfx = base + '├─ '
        end_pfx = base + '└─ '

    def _step_tag(idx):
        return f'[{idx:02d}/{total_actions:02d}]'

    logger.info('%sTaskStart: %s  steps=%d', start_pfx, context_key, total_actions)

    try:
        for idx, act in enumerate(steps, start=1):
            name = act[0]
            params = act[1:]
            tag = _step_tag(idx)

            # check for task cancellation (silent unless triggered)
            if ui._task_cancel_event.is_set():
                logger.info('%s%s %-9s → cancelled', sp, tag, name)
                fail_count += 1
                result = False
                break

            if name == 'locate':
                tpl_path = Path(context_key)
                if src_dir:
                    tpl_path = src_dir.joinpath(context_key)

                max_attempts = params[0]
                res = None
                for attempt in range(1, max_attempts + 1):
                    time.sleep(params[1] / 1000.0)
                    res = ui.match_template(tpl_path, threshold=0.8, map_to_device=True, debug=False)
                    if res is not None:
                        break
                    logger.debug('%s%s locate attempt %d/%d failed for %s', sp, tag, attempt, max_attempts, tpl_path.name)

                if res is None:
                    logger.info('%s%s %-9s %-20s → failed  reason=template_not_found', sp, tag, name, tpl_path.name)
                    fail_count += 1
                    result = False
                    break
                dev_x, dev_y, elapsed, score = res
                loc_coord = (dev_x, dev_y)
                logger.info('%s%s %-9s %-20s → ok  dev=(%d,%d)  score=%.4f  %.2fs', sp, tag, name, tpl_path.name, dev_x, dev_y, score, elapsed)
                ok_count += 1

            elif name == 'enter':
                if loc_coord is None:
                    logger.info('%s%s %-9s → skipped  reason=no_loc_coord', sp, tag, name)
                    skip_count += 1
                    continue
                dev_x, dev_y = loc_coord
                jx, jy = _apply_jitter(dev_x, dev_y, *params)
                if ui.controller is not None:
                    try:
                        ui.controller.tap(jx, jy)
                        logger.info('%s%s %-9s tap=(%d,%d)  → ok', sp, tag, name, jx, jy)
                        ok_count += 1
                    except Exception:
                        logger.info('%s%s %-9s tap=(%d,%d)  → failed  reason=tap_exception', sp, tag, name, jx, jy)
                        fail_count += 1
                else:
                    logger.info('%s%s %-9s → failed  reason=no_controller', sp, tag, name)
                    fail_count += 1

            elif name == 'sleep':
                ms = int(params[0])
                total = ms / 1000.0
                step = 0.05
                elapsed = 0.0
                interrupted = False
                while elapsed < total:
                    if ui._task_cancel_event.is_set():
                        interrupted = True
                        break
                    to_sleep = min(step, total - elapsed)
                    time.sleep(to_sleep)
                    elapsed += to_sleep
                if interrupted:
                    logger.info('%s%s %-9s %dms → interrupted', sp, tag, name, ms)
                    fail_count += 1
                    result = False
                    break
                logger.info('%s%s %-9s %dms → ok', sp, tag, name, ms)
                ok_count += 1

            elif name == 'click':
                x0, y0 = int(params[0][0]), int(params[0][1])
                jx, jy = _apply_jitter(x0, y0, *params[1:])
                if ui.controller is not None:
                    try:
                        ui.controller.tap(jx, jy)
                        logger.info('%s%s %-9s (%d,%d) tap=(%d,%d)  → ok', sp, tag, name, x0, y0, jx, jy)
                        ok_count += 1
                    except Exception:
                        logger.info('%s%s %-9s (%d,%d) tap=(%d,%d)  → failed  reason=tap_exception', sp, tag, name, x0, y0, jx, jy)
                        fail_count += 1
                else:
                    logger.info('%s%s %-9s (%d,%d) → failed  reason=no_controller', sp, tag, name, x0, y0)
                    fail_count += 1

            elif name == 'match':
                if len(params) < 1:
                    logger.info('%s%s %-9s → skipped  reason=missing_target', sp, tag, name)
                    skip_count += 1
                    continue
                target_key = params[0]
                # Mode: "required"=fail stops, "optional"=fail continues, "absent"=success stops
                mode = str(params[1]) if len(params) >= 2 else 'required'

                action_list = getattr(cfg, 'action_list', {}).get(target_key)
                if not action_list:
                    logger.info('%s%s %-9s %-20s → skipped  reason=no_action_list', sp, tag, name, target_key)
                    skip_count += 1
                    continue
                res = _execute_action(ui, action_list, target_key, src_dir, cfg, depth + 1)

                if mode == 'absent':
                    # Inverse: success → stop, fail → continue
                    if res:
                        logger.info('%s%s %-9s %-20s → matched', sp, tag, name, target_key)
                        fail_count += 1
                        result = False
                        break
                    logger.info('%s%s %-9s %-20s not_found  → ok', sp, tag, name, target_key)
                    ok_count += 1
                else:
                    # Normal: success → continue
                    if res:
                        logger.info('%s%s %-9s %-20s → ok', sp, tag, name, target_key)
                        ok_count += 1
                    elif mode == 'optional':
                        # Fail but continue
                        logger.info('%s%s %-9s %-20s → ok_with_errors', sp, tag, name, target_key)
                        ok_count += 1
                    else:
                        # 'required': fail stops
                        logger.info('%s%s %-9s %-20s → failed  reason=nested_failed', sp, tag, name, target_key)
                        fail_count += 1
                        result = False
                        break

            else:
                logger.info('%s%s %-9s %s → unknown', sp, tag, name, params)
                skip_count += 1
    finally:
        elapsed_total = time.perf_counter() - start_time
        logger.info('%sTaskEnd:   %s  result=%s  ok=%d skip=%d fail=%d  elapsed=%.1fs',
                    end_pfx, context_key, result, ok_count, skip_count, fail_count, elapsed_total)
    return result


def handle_task(ui, task_name: str, task: dict, action_list: dict) -> None:
    """Dispatch task based on task['action'].

    Supports 'match_list' action which iterates images in task['img_path']
    and executes steps defined in `config.action_list`.
    Supports 'simple' action which executes steps for task['name'] once.
    """
    action = task.get('action')
    logger.info('handle_task: %s -> action=%s', task_name, action)

    cfg = importlib.import_module('config')

    if action == 'simple':
        name = task.get('name')
        steps = action_list.get(name)
        if not steps:
            logger.warning('No action list for simple task: %s', name)
            return
        img_path = task.get('img_path')
        src_dir = Path(img_path) if img_path else None
        _execute_action(ui, steps, name, src_dir, cfg)

    elif action == 'match_list':
        img_path = task.get('img_path')
        src_dir = Path(img_path) if img_path else None

        # Iterate over the list repeatedly until a full pass produces no truthy results
        any_active = True
        while any_active:
            any_active = False
            for key in task.get('match_list', []):
                if action_list.get(key, None) is None:
                    logger.warning('No action list for key: %s', key)
                    continue
                result = _execute_action(ui, action_list[key], key, src_dir, cfg)
                if result:
                    any_active = True
    else:
        logger.info('No implementation for action: %s', action)
