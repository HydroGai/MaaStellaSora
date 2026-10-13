from dataclasses import dataclass
from typing import Any

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.custom_action import CustomAction
from maa.context import Context
from maa.define import Rect

from utils import logger as logger_module
logger = logger_module.get_logger("operation")


RESOURCE_RECOGNITION_NODE = "猎影合围_识别资源数量"


def _get_keep_count(context: Context, key: str) -> int | None:
    node_data = context.get_node_data(RESOURCE_RECOGNITION_NODE)
    if not isinstance(node_data, dict):
        return None
    attach = node_data.get("attach", {})
    if not isinstance(attach, dict):
        return None
    count = attach.get(key, 0)
    if type(count) is not int or count < 0:
        logger.warning("保留数量配置无效：%s", key)
        return None
    return count

def _get_resource_count(context: Context, image: np.ndarray) -> int | None:
    reco_result = context.run_recognition(RESOURCE_RECOGNITION_NODE, image)
    if reco_result and reco_result.hit and reco_result.best_result:
        try:
            return int(reco_result.best_result.text.split("/")[0])
        except (AttributeError, TypeError, ValueError):
            logger.debug("资源数量 OCR 结果无效")
    return None

def _locate_track_button(context: Context, image: np.ndarray) -> list[Rect] | None:
    reco_result = context.run_recognition("猎影合围_追踪目标_追踪按钮", image)
    if reco_result and reco_result.hit:
        return [result.box for result in (reco_result.filtered_results or [])]
    return None

def _locate_hunt_again_button(context: Context, image: np.ndarray) -> Rect | None:
    reco_result = context.run_recognition("__猎影合围_追踪目标_再次讨伐按钮", image)
    if reco_result and reco_result.hit and reco_result.best_result:
        return reco_result.best_result.box
    return None


@AgentServer.custom_recognition("enough_tracking_permit_recognition")
class EnoughTrackingPermitRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查追踪委托书是否大于保留数量及追踪按钮是否可用。"""
        track_button_boxes = _locate_track_button(context, argv.image)
        if not track_button_boxes or len(track_button_boxes) != 2:
            logger.debug("未找到2个追踪按钮")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        keep_count = _get_keep_count(context, "tracking_permit_keep_count")
        count = _get_resource_count(context, argv.image)
        if count is None or keep_count is None:
            logger.debug("未识别到追踪委托书数量")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        usable = count - keep_count
        if count <= 0 or usable <= 0:
            logger.debug(f"追踪委托书数量{count}小于等于保留数量{keep_count}或数量为0")
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        
        logger.info(f"追踪委托书数量{count}大于保留数量{keep_count}，开始追踪")
        quick_track_cost = min(5, max(2, count))
        if usable >= quick_track_cost:
            track_button_box = track_button_boxes[0]
        else:
            track_button_box = track_button_boxes[-1]
        return CustomRecognition.AnalyzeResult(box=track_button_box, detail={})

@AgentServer.custom_recognition("lack_of_tracking_permit_recognition")
class LackOfTrackingPermitRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查追踪委托书是否小于保留数量及再次讨伐按钮、追踪按钮是否可用。"""
        keep_count = _get_keep_count(context, "tracking_permit_keep_count")
        count = _get_resource_count(context, argv.image)
        logger.debug(f"结束追踪模块前检查：追踪委托书数量{count}，保留数量{keep_count}")
        if count is None or keep_count is None or (count > keep_count and count > 0):
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        hunt_again_button_box = _locate_hunt_again_button(context, argv.image)
        track_button_box = _locate_track_button(context, argv.image)
        if not hunt_again_button_box and not track_button_box:
            logger.debug("未找到再次讨伐按钮和追踪按钮，继续追踪")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        logger.debug(f"追踪委托书数量{count}小于保留数量{keep_count}，且找到再次讨伐按钮或追踪按钮，退出追踪")
        return CustomRecognition.AnalyzeResult(box=(1, 1, 1, 1), detail={})

@AgentServer.custom_recognition("enough_hunt_license_recognition")
class EnoughHuntLicenseRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查围猎许可证是否大于保留数量及协助按钮是否可用。"""
        coop_button_box = self._locate_coop_button(context, argv.image)
        if not coop_button_box:
            logger.debug("未找到协助按钮")
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        keep_count = _get_keep_count(context, "hunt_license_keep_count")
        count = _get_resource_count(context, argv.image)
        if count is None or keep_count is None or count <= keep_count or count <= 0:
            logger.debug(f"围猎许可证数量{count}小于等于保留数量{keep_count}或数量为0")
            return CustomRecognition.AnalyzeResult(box=None, detail={})
        
        logger.info(f"围猎许可证数量{count}大于保留数量{keep_count}，开始协助讨伐")
        return CustomRecognition.AnalyzeResult(box=coop_button_box, detail={})

    @staticmethod
    def _locate_coop_button(context: Context, image: np.ndarray) -> Rect | None:
        reco_result = context.run_recognition("猎影合围_协助_协助讨伐按钮", image)
        if reco_result and reco_result.hit and reco_result.best_result:
            return reco_result.best_result.box
        return None

@AgentServer.custom_recognition("lack_of_hunt_license_recognition")
class LackOfHuntLicenseRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:
        """检查围猎许可证是否小于保留数量。"""
        keep_count = _get_keep_count(context, "hunt_license_keep_count")
        count = _get_resource_count(context, argv.image)
        logger.debug(f"结束协助模块前检查：围猎许可证数量{count}，保留数量{keep_count}")
        if count is None or keep_count is None or (count > keep_count and count > 0):
            return CustomRecognition.AnalyzeResult(box=None, detail={})

        return CustomRecognition.AnalyzeResult(box=(1, 1, 1, 1), detail={})

@dataclass
class FreeCoopState:
    target_count: int = 5
    completed_count: int = 0
    current_is_free: bool = False
    logged_finish: bool = False


_context_coop_states: dict[Any, FreeCoopState] = {}

def _get_free_coop_state(context: Context, argv: Any = None) -> FreeCoopState:
    """获取当前任务上下文绑定的无偿协助运行状态。"""
    key = getattr(getattr(argv, "task_detail", None), "task_id", None)
    if key is None and hasattr(context, "_handle") and hasattr(context._handle, "value"):
        key = context._handle.value
    if key is None:
        key = id(context)

    if key not in _context_coop_states:
        if len(_context_coop_states) > 50:
            for old_key in list(_context_coop_states.keys())[:-20]:
                _context_coop_states.pop(old_key, None)
        _context_coop_states[key] = FreeCoopState()
    return _context_coop_states[key]


@AgentServer.custom_action("free_coop_init")
class FreeCoopInit(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        node_data = context.get_node_data("猎影合围无偿协助_参数配置")
        target = 5
        if isinstance(node_data, dict):
            attach = node_data.get("attach", {})
            if isinstance(attach, dict):
                val = attach.get("free_coop_target_count", 5)
                try:
                    target = int(val)
                except (ValueError, TypeError):
                    target = 5

        state = _get_free_coop_state(context, argv)
        state.target_count = max(1, target)
        state.completed_count = 0
        state.current_is_free = False
        state.logged_finish = False
        logger.info(f"猎影合围无偿协助任务初始化，目标次数: {state.target_count}")
        return True


@AgentServer.custom_recognition("free_coop_finish_recognition")
class FreeCoopFinishRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:  # 检查无偿协助是否已经达到目标次数
        state = _get_free_coop_state(context, argv)
        if state.completed_count >= state.target_count:
            if not state.logged_finish:
                logger.info(f"无偿协助已达到目标次数 ({state.completed_count}/{state.target_count})，准备退出")
                state.logged_finish = True
            return CustomRecognition.AnalyzeResult(box=(1, 1, 1, 1), detail={})
        return CustomRecognition.AnalyzeResult(box=None, detail={})  # 未达到目标次数，继续追踪


@AgentServer.custom_recognition("free_coop_button_recognition")
class FreeCoopButtonRecognition(CustomRecognition):
    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult:  # 识别协助按钮。若有许可证先消耗许可证；若为无偿协助且未满次数则执行协助
        state = _get_free_coop_state(context, argv)
        if state.completed_count >= state.target_count:
            return CustomRecognition.AnalyzeResult(box=None, detail={})  # 已达目标次数，不执行协助

        reco_result = context.run_recognition("猎影合围_协助_协助讨伐按钮", argv.image)
        if not (reco_result and reco_result.hit and reco_result.best_result):
            # 若基础配置未匹配到（如特定区服未覆盖无偿标签），使用包含无偿协助的预期文本进行覆盖重试
            reco_result = context.run_recognition(
                "猎影合围_协助_协助讨伐按钮",
                argv.image,
                pipeline_override={
                    "猎影合围_协助_协助讨伐按钮": {
                        "recognition": {
                            "param": {
                                "expected": [
                                    "协助讨伐",
                                    "无偿协助",
                                    "協助討伐",
                                    "無償協助"
                                ]
                            }
                        }
                    }
                }
            )

        if not (reco_result and reco_result.hit and reco_result.best_result):
            logger.debug("未找到协助讨伐/无偿协助按钮")
            return CustomRecognition.AnalyzeResult(box=None, detail={})  # 未找到协助按钮，继续追踪

        btn_box = reco_result.best_result.box
        btn_text = reco_result.best_result.text or ""
        count = _get_resource_count(context, argv.image)
        # 判断是否为无偿协助：
        # 如果许可证数量为0，或者按钮文字包含"無償"，则判定为无偿协助 除台服外其他区服未知(日服或为"救援参加")
        is_free = False
        if count == 0 or "無償" in btn_text or "无偿" in btn_text:  # 此处判断需要完善各区服文本以加强判断
            is_free = True
        elif count is not None and count > 0:
            is_free = False
        else:
            is_free = ("無償" in btn_text or "无偿" in btn_text)  # 此处判断需要完善各区服文本以加强判断

        state.current_is_free = is_free
        if is_free:
            logger.info(f"开始第 {state.completed_count + 1} 次无偿协助 (目标: {state.target_count} 次)")
        else:
            logger.info(f"许可证剩余 {count}，先进行正常协助消耗许可证")

        return CustomRecognition.AnalyzeResult(box=btn_box, detail={})


@AgentServer.custom_action("free_coop_battle_finish")
class FreeCoopBattleFinish(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        state = _get_free_coop_state(context, argv)
        if state.current_is_free:
            state.completed_count += 1
            logger.info(f"完成一次无偿协助，当前进度: {state.completed_count}/{state.target_count}")
            state.current_is_free = False
        try:
            context.tasker.controller.post_click(640, 360).wait()  # 此处点击屏幕，若失败可能导致任务中断，需要进一步测试以修复
        except Exception as e:
            logger.warning(f"战斗结束点击屏幕失败: {e}")  # 此处抛出异常，任务可能因此中断，需要进一步测试以修复
        return True
