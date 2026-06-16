from log import get_logger
import os
import time
from pathlib import Path
from typing import Optional

import importlib

from utils import _apply_jitter

logger = get_logger(__name__)


def _execute_action(ui, steps, context_key, src_dir: Path, cfg) -> bool:
    """Execute a list of step tuples for the given context key.

    Returns True if execution completed and may be repeated; returns False
    if a 'locate' action failed (meaning stop processing this entry and move on).
    """
    loc_coord = None
    total_actions = len(steps) if steps is not None else 0
    for idx, act in enumerate(steps, start=1):
        name = act[0]
        params = act[1:]

        # default result
        result = 'ok'
        result_reason = 'none'

        # check for task cancellation
        try:
            if getattr(ui, '_task_cancel_event', None) is not None and ui._task_cancel_event.is_set():
                result = 'cancelled'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s', context_key, idx, name, params, result)
                return False
        except Exception:
            result = 'error_check_cancel'
            result_reason = 'exception checking cancel event'
            logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)
            return False

        if name == 'locate':
            # resolve template path (safe if src_dir is None)
            tpl_path = Path(context_key)
            if src_dir:
                tpl_path = src_dir.joinpath(context_key)

            # try up to param[0] attempts before giving up
            max_attempts = params[0]
            res = None
            for attempt in range(1, max_attempts + 1):
                # on retries wait 100ms (check cancel during wait)
                time.sleep(params[1] / 1000.0)
                res = ui.match_template(tpl_path, threshold=0.8, map_to_device=True, debug=False)
                if res is not None:
                    break
                logger.debug('locate attempt %d/%d failed for %s', attempt, max_attempts, tpl_path)

            if res is None:
                result = 'failed'
                result_reason = 'template_not_found'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, [str(tpl_path)], result, result_reason)
                return False
            dev_x, dev_y, elapsed = res
            loc_coord = (dev_x, dev_y)
            logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s', context_key, idx, name, [dev_x, dev_y, elapsed], result)

        elif name == 'enter':
            if loc_coord is None:
                result = 'skipped'
                result_reason = 'no_loc_coord'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)
                continue
            dev_x, dev_y = loc_coord
            jx, jy = _apply_jitter(dev_x, dev_y, *params)
            if hasattr(ui, 'controller') and ui.controller is not None:
                try:
                    ui.controller.tap(jx, jy)
                except Exception:
                    result = 'failed'
                    result_reason = 'tap_exception'
            else:
                result = 'failed'
                result_reason = 'no_controller'
            # log the coordinates actually passed to tap (jx, jy)
            logger.info('ActionExec: key=%s idx=%d step=%s tap_coords=%s params=%s result=%s reason=%s', context_key, idx, name, [jx, jy], params, result, result_reason)

        elif name == 'sleep':
            ms = int(params[0])
            total = ms / 1000.0
            step = 0.05
            elapsed = 0.0
            interrupted = False
            while elapsed < total:
                try:
                    if getattr(ui, '_task_cancel_event', None) is not None and ui._task_cancel_event.is_set():
                        interrupted = True
                        break
                except Exception:
                    interrupted = True
                    break
                to_sleep = min(step, total - elapsed)
                time.sleep(to_sleep)
                elapsed += to_sleep
            if interrupted:
                result = 'interrupted'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s', context_key, idx, name, [ms], result)
                return False
            logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s', context_key, idx, name, [ms], result)

        elif name == 'click':
            if not params:
                result = 'skipped'
                result_reason = 'missing_coords'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)
                continue
            coords = params[0]
            if not (isinstance(coords, (list, tuple)) and len(coords) >= 2):
                result = 'skipped'
                result_reason = 'invalid_coords'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, coords, result, result_reason)
                continue
            try:
                x0, y0 = int(coords[0]), int(coords[1])
            except Exception:
                result = 'skipped'
                result_reason = 'coord_type_error'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, coords, result, result_reason)
                continue
            jx, jy = _apply_jitter(x0, y0, *params[1:])
            if hasattr(ui, 'controller') and ui.controller is not None:
                try:
                    ui.controller.tap(jx, jy)
                except Exception:
                    result = 'failed'
                    result_reason = 'tap_exception'
            else:
                result = 'failed'
                result_reason = 'no_controller'
            # log the coordinates actually passed to tap (jx, jy)
            logger.info('ActionExec: key=%s idx=%d step=%s tap_coords=%s params=%s result=%s reason=%s', context_key, idx, name, [jx, jy], params[1:], result, result_reason)

        elif name == 'match':
            if len(params) < 1:
                result = 'skipped'
                result_reason = 'missing_target'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)
                continue
            target_key = params[0]
            continue_on_fail = False
            if len(params) >= 2:
                try:
                    continue_on_fail = bool(params[1])
                except Exception:
                    continue_on_fail = False
            action_list = getattr(cfg, 'action_list', {}).get(target_key)
            if not action_list:
                result = 'skipped'
                result_reason = 'no_action_list'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)
                continue
            res = _execute_action(ui, action_list, target_key, src_dir, cfg)
            if not res and not continue_on_fail:
                result = 'failed'
                result_reason = 'nested_failed'
                logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)
                return False
            result = 'ok' if res else 'ok_with_errors'
            logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s', context_key, idx, name, params, result)

        else:
            result = 'unknown'
            result_reason = 'unknown_action'
            logger.info('ActionExec: key=%s idx=%d step=%s params=%s result=%s reason=%s', context_key, idx, name, params, result, result_reason)

    # completed this pass; return True to indicate caller may repeat
    logger.info('Action pass complete: context=%s', context_key)
    return True


def handle_task(ui, task_name: str, task: dict, action_list: dict) -> None:
    """Dispatch task based on task['action'].

    Supports 'match_list' action which iterates images in task['img_path']
    and executes steps defined in `config.action_list`.
    """
    action = task.get('action')
    logger.info('handle_task: %s -> action=%s', task_name, action)

    if action != 'match_list':
        logger.info('No implementation for action: %s', action)
        return

    # load config module to access action_list
    try:
        cfg = importlib.import_module('config')
    except Exception:
        logger.exception('Failed to import config')
        return

    img_path = task.get('img_path')
    src_dir = Path(img_path) if img_path else None

    # Iterate over the list repeatedly until a full pass produces no truthy results
    # _execute_action returns True to indicate the action pass completed (may be
    # repeated); returns False to indicate 'locate' failed / processing should stop
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
