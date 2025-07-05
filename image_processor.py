import numpy as np
from PIL import Image
import base64
import io
import json
from typing import List, Dict, Any, Tuple, Optional
import math
import os

class ImageProcessor:
    """处理图片和数组之间的转换，支持分卷处理"""
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB 默认分块大小
        self.chunk_size = chunk_size
        self.max_dimension = 2048  # 最大图片尺寸限制
    
    def array_to_image(self, data: List[List[List[int]]], format: str = "PNG") -> str:
        """
        将3D数组转换为图片的base64编码
        
        Args:
            data: 3D数组 [height, width, channels]
            format: 图片格式 (PNG, JPEG等)
            
        Returns:
            base64编码的图片字符串
        """
        try:
            # 转换为numpy数组
            np_array = np.array(data, dtype=np.uint8)
            
            # 检查数组维度
            if len(np_array.shape) == 2:
                # 灰度图
                mode = 'L'
            elif len(np_array.shape) == 3:
                if np_array.shape[2] == 1:
                    # 单通道
                    np_array = np_array.squeeze(axis=2)
                    mode = 'L'
                elif np_array.shape[2] == 3:
                    # RGB
                    mode = 'RGB'
                elif np_array.shape[2] == 4:
                    # RGBA
                    mode = 'RGBA'
                else:
                    raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
            else:
                raise ValueError(f"不支持的数组维度: {np_array.shape}")
            
            # 创建PIL图片
            image = Image.fromarray(np_array, mode=mode)
            
            # 转换为base64
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            raise Exception(f"数组转图片失败: {str(e)}")
    
    def image_to_array(self, image_base64: str) -> List[List[List[int]]]:
        """
        将base64编码的图片转换为3D数组
        
        Args:
            image_base64: base64编码的图片字符串
            
        Returns:
            3D数组 [height, width, channels]
        """
        try:
            # 解码base64
            image_data = base64.b64decode(image_base64)
            
            # 创建PIL图片
            image = Image.open(io.BytesIO(image_data))
            
            # 确保图片模式正确
            if image.mode not in ['L', 'RGB', 'RGBA']:
                # 转换其他模式到RGB
                image = image.convert('RGB')
            
            # 转换为numpy数组并确保类型为uint8
            np_array = np.array(image, dtype=np.uint8)
            
            # 确保是3D数组
            if len(np_array.shape) == 2:
                # 灰度图，添加通道维度
                np_array = np.expand_dims(np_array, axis=2)
            
            # 验证数值范围
            if np_array.min() < 0 or np_array.max() > 255:
                raise ValueError("图片数组值必须在0-255范围内")
            
            # 转换为Python列表
            return np_array.tolist()
            
        except Exception as e:
            raise Exception(f"图片转数组失败: {str(e)}")
    
    def create_chunked_image(self, data: List[List[List[int]]], format: str = "PNG") -> Dict[str, Any]:
        """
        创建分卷图片数据
        
        Args:
            data: 大图片的3D数组
            format: 图片格式
            
        Returns:
            包含分卷信息的字典
        """
        try:
            np_array = np.array(data, dtype=np.uint8)
            height, width = np_array.shape[:2]
            
            # 检查是否需要分卷
            total_pixels = height * width
            if len(np_array.shape) == 3:
                total_pixels *= np_array.shape[2]
            
            if total_pixels * 4 <= self.chunk_size:  # 不需要分卷
                image_base64 = self.array_to_image(data, format)
                return {
                    "is_chunked": False,
                    "data": image_base64,
                    "original_shape": [height, width] + ([] if len(np_array.shape) == 2 else [np_array.shape[2]]),
                    "format": format
                }
            
            # 计算分块大小
            chunk_height = min(height, int(math.sqrt(self.chunk_size / (width * (np_array.shape[2] if len(np_array.shape) == 3 else 1)))))
            chunk_height = max(1, chunk_height)
            
            chunks = []
            chunk_info = []
            
            for start_row in range(0, height, chunk_height):
                end_row = min(start_row + chunk_height, height)
                chunk_data = np_array[start_row:end_row]
                
                # 转换为base64
                chunk_base64 = self.array_to_image(chunk_data.tolist(), format)
                chunks.append(chunk_base64)
                
                chunk_info.append({
                    "start_row": start_row,
                    "end_row": end_row,
                    "shape": list(chunk_data.shape)
                })
            
            return {
                "is_chunked": True,
                "chunks": chunks,
                "chunk_info": chunk_info,
                "original_shape": list(np_array.shape),
                "total_chunks": len(chunks),
                "format": format
            }
            
        except Exception as e:
            raise Exception(f"创建分卷图片失败: {str(e)}")
    
    def parse_chunked_image(self, chunked_data: Dict[str, Any]) -> List[List[List[int]]]:
        """
        解析分卷图片数据
        
        Args:
            chunked_data: 分卷数据字典
            
        Returns:
            完整的3D数组
        """
        try:
            if not chunked_data.get("is_chunked", False):
                # 不是分卷数据，直接解析
                return self.image_to_array(chunked_data["data"])
            
            # 解析分卷数据
            original_shape = chunked_data["original_shape"]
            chunks = chunked_data["chunks"]
            chunk_info = chunked_data["chunk_info"]
            
            # 创建结果数组
            if len(original_shape) == 2:
                result = np.zeros(original_shape, dtype=np.uint8)
            else:
                result = np.zeros(original_shape, dtype=np.uint8)
            
            # 组合分块
            for i, (chunk_base64, info) in enumerate(zip(chunks, chunk_info)):
                chunk_array = np.array(self.image_to_array(chunk_base64), dtype=np.uint8)
                
                # 如果是灰度图但原图有通道维度
                if len(original_shape) == 3 and len(chunk_array.shape) == 3 and chunk_array.shape[2] == 1:
                    chunk_array = chunk_array.squeeze(axis=2)
                elif len(original_shape) == 2 and len(chunk_array.shape) == 3:
                    chunk_array = chunk_array.squeeze(axis=2)
                
                start_row = info["start_row"]
                end_row = info["end_row"]
                
                if len(original_shape) == 2:
                    result[start_row:end_row] = chunk_array
                else:
                    result[start_row:end_row] = chunk_array
            
            return result.tolist()
            
        except Exception as e:
            raise Exception(f"解析分卷图片失败: {str(e)}")
    
    def get_image_info(self, image_base64: str) -> Dict[str, Any]:
        """
        获取图片信息
        
        Args:
            image_base64: base64编码的图片
            
        Returns:
            图片信息字典
        """
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(image_data)
            }
        except Exception as e:
            raise Exception(f"获取图片信息失败: {str(e)}")
    
    def file_to_array(self, file_path: str) -> List[List[List[int]]]:
        """
        从文件路径读取图片并转换为3D数组
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            3D数组 [height, width, channels]
        """
        try:
            # 读取图片文件
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # 转换为base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 使用现有方法转换为数组
            return self.image_to_array(image_base64)
            
        except Exception as e:
            raise Exception(f"文件转数组失败: {str(e)}")
    
    def save_base64_to_file(self, image_base64: str, file_path: str, format: str = None) -> str:
        """
        将base64编码的图片保存为文件
        
        Args:
            image_base64: base64编码的图片字符串
            file_path: 保存的文件路径
            format: 图片格式（可选，如果不指定则从文件扩展名推断）
            
        Returns:
            保存的文件的绝对路径
        """
        try:
            # 解码base64
            image_data = base64.b64decode(image_base64)
            
            # 创建PIL图片对象
            image = Image.open(io.BytesIO(image_data))
            
            # 如果没有指定格式，从文件扩展名推断
            if format is None:
                format = os.path.splitext(file_path)[1].lstrip('.')
                if not format:
                    format = 'PNG'  # 默认使用PNG
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存图片
            image.save(file_path, format=format.upper())
            
            return os.path.abspath(file_path)
            
        except Exception as e:
            raise Exception(f"保存图片文件失败: {str(e)}")
    
    def save_array_to_file(self, array_data: List[List[List[int]]], file_path: str, format: str = "PNG") -> str:
        """
        将3D数组保存为图片文件
        
        Args:
            array_data: 3D数组 [height, width, channels]
            file_path: 保存的文件路径
            format: 图片格式（默认PNG）
            
        Returns:
            保存的文件的绝对路径
        """
        try:
            # 转换为numpy数组并确保类型为uint8
            np_array = np.array(array_data, dtype=np.uint8)
            
            # 检查数组维度
            if len(np_array.shape) == 2:
                mode = 'L'
            elif len(np_array.shape) == 3:
                if np_array.shape[2] == 1:
                    np_array = np_array.squeeze(axis=2)
                    mode = 'L'
                elif np_array.shape[2] == 3:
                    mode = 'RGB'
                elif np_array.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    raise ValueError(f"不支持的通道数: {np_array.shape[2]}")
            else:
                raise ValueError(f"不支持的数组维度: {np_array.shape}")
            
            # 验证数值范围
            if np_array.min() < 0 or np_array.max() > 255:
                raise ValueError("图片数组值必须在0-255范围内")
            
            # 创建PIL图片
            image = Image.fromarray(np_array, mode=mode)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存图片
            image.save(file_path, format=format)
            
            return os.path.abspath(file_path)
            
        except Exception as e:
            raise Exception(f"保存数组到文件失败: {str(e)}")
    
    def save_array(self, array_data: List[List[List[int]]], file_path: str, format: str = "npy") -> str:
        """
        直接保存数组数据到文件
        
        Args:
            array_data: 3D数组 [height, width, channels]
            file_path: 保存的文件路径
            format: 保存格式，支持 "npy"（numpy格式）或 "json"
            
        Returns:
            保存的文件的绝对路径
        """
        try:
            # 转换为numpy数组
            np_array = np.array(array_data, dtype=np.uint8)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if format.lower() == "npy":
                # 如果没有.npy后缀，添加它
                if not file_path.lower().endswith('.npy'):
                    file_path += '.npy'
                # 保存为numpy格式
                np.save(file_path, np_array)
            elif format.lower() == "json":
                # 如果没有.json后缀，添加它
                if not file_path.lower().endswith('.json'):
                    file_path += '.json'
                # 保存为JSON格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'shape': np_array.shape,
                        'data': np_array.tolist()
                    }, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
            
            return os.path.abspath(file_path)
            
        except Exception as e:
            raise Exception(f"保存数组数据失败: {str(e)}")
    
    def load_array(self, file_path: str) -> List[List[List[int]]]:
        """
        从文件加载数组数据
        
        Args:
            file_path: 数组数据文件路径
            
        Returns:
            3D数组 [height, width, channels]
        """
        try:
            if file_path.lower().endswith('.npy'):
                # 从numpy格式加载
                np_array = np.load(file_path)
            elif file_path.lower().endswith('.json'):
                # 从JSON格式加载
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    np_array = np.array(data['data'], dtype=np.uint8)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 验证数组
            if not (2 <= len(np_array.shape) <= 3):
                raise ValueError(f"不支持的数组维度: {np_array.shape}")
            
            # 确保是3D数组
            if len(np_array.shape) == 2:
                np_array = np.expand_dims(np_array, axis=2)
            
            return np_array.tolist()
            
        except Exception as e:
            raise Exception(f"加载数组数据失败: {str(e)}") 