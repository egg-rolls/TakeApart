"""TakeApart MCP Server — AI 视觉拆解工具."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tools.detect import detect_elements
from .tools.segment import segment_element

logger = logging.getLogger("takeapart")


# ─── Tool Definitions ────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="detect_elements",
        description=(
            "轻量级元素检测（纯传统CV，无需GPU）。通过 Canny 边缘检测 + 轮廓分析，"
            "快速发现图片中所有独立元素的大致位置和边界框。"
            "适用于设计稿、海报、UI截图、PPT等图片中分散元素的快速发现。"
            "可通过 output_path 参数输出标注了所有检测框的可视化图片。"
            "注意：这是无视觉能力AI的 fallback 工具，精度有限。"
            "如需精确分割，请使用 segment_element。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图片文件路径",
                },
                "output_path": {
                    "type": "string",
                    "description": "标注可视化图片的输出路径（留空则不输出）",
                    "default": "",
                },
                "min_area_ratio": {
                    "type": "number",
                    "description": "最小面积比例 (0-1)，过滤噪声。默认 0.002",
                    "default": 0.002,
                    "minimum": 0,
                    "maximum": 1,
                },
                "blur_kernel": {
                    "type": "integer",
                    "description": "高斯模糊核大小（奇数，0=关闭）。默认 5",
                    "default": 5,
                },
                "canny_low": {
                    "type": "integer",
                    "description": "Canny 边缘检测低阈值。默认 50",
                    "default": 50,
                },
                "canny_high": {
                    "type": "integer",
                    "description": "Canny 边缘检测高阈值。默认 150",
                    "default": 150,
                },
            },
            "required": ["image_path"],
        },
    ),
    Tool(
        name="segment_element",
        description=(
            "精确分割工具（SAM + Canny 边缘精修）。"
            "PS 式操作：传入框选坐标，算法自动找到框内最清晰边缘的完整元素，"
            "生成硬边缘的分割结果。支持批量处理多个框。"
            "输出模式可选：mask（返回 base64 mask）、png（导出透明 PNG）、both。"
            "推荐工作流：先用视觉AI分析图片获取元素坐标，再调用此工具精确分割。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图片文件路径",
                },
                "boxes": {
                    "type": "array",
                    "description": "边界框列表，每个框含 x1,y1,x2,y2 坐标和可选 label",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x1": {"type": "integer", "description": "左上角 X"},
                            "y1": {"type": "integer", "description": "左上角 Y"},
                            "x2": {"type": "integer", "description": "右下角 X"},
                            "y2": {"type": "integer", "description": "右下角 Y"},
                            "label": {"type": "string", "description": "元素标签", "default": ""},
                        },
                        "required": ["x1", "y1", "x2", "y2"],
                    },
                },
                "output_mode": {
                    "type": "string",
                    "description": "输出模式：mask(返回base64) | png(导出文件) | both(两者都)",
                    "enum": ["mask", "png", "both"],
                    "default": "png",
                },
                "output_dir": {
                    "type": "string",
                    "description": "PNG 输出目录（留空则使用默认 output/ 目录）",
                    "default": "",
                },
                "padding": {
                    "type": "integer",
                    "description": "元素周围的内边距像素数。默认 10",
                    "default": 10,
                },
                "crop": {
                    "type": "boolean",
                    "description": "是否裁剪到元素边界。默认 true",
                    "default": True,
                },
                "edge_refine": {
                    "type": "boolean",
                    "description": "是否启用 Canny 边缘精修（关闭可加速）。默认 true",
                    "default": True,
                },
            },
            "required": ["image_path", "boxes"],
        },
    ),
]


# ─── Tool Handlers ───────────────────────────────────────────────

def _run_detect(args: dict) -> str:
    result = detect_elements(
        image_path=args["image_path"],
        output_path=args.get("output_path", ""),
        min_area_ratio=args.get("min_area_ratio", 0.002),
        blur_kernel=args.get("blur_kernel", 5),
        canny_low=args.get("canny_low", 50),
        canny_high=args.get("canny_high", 150),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def _run_segment(args: dict) -> str:
    result = segment_element(
        image_path=args["image_path"],
        boxes=args["boxes"],
        output_mode=args.get("output_mode", "png"),
        output_dir=args.get("output_dir", ""),
        padding=args.get("padding", 10),
        crop=args.get("crop", True),
        edge_refine=args.get("edge_refine", True),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


TOOL_HANDLERS = {
    "detect_elements": _run_detect,
    "segment_element": _run_segment,
}


# ─── MCP Server ──────────────────────────────────────────────────

def create_server() -> Server:
    server = Server("takeapart")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            result = handler(arguments)
            return [TextContent(type="text", text=result)]
        except FileNotFoundError as e:
            return [TextContent(type="text", text=f"文件未找到: {e}")]
        except Exception as e:
            logger.exception("Tool error: %s", name)
            return [TextContent(type="text", text=f"工具执行错误: {type(e).__name__}: {e}")]

    return server


async def run_server() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting TakeApart MCP Server...")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
