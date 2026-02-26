"""
此文件是训练脚本的多模式启动器，用于在不同的计算环境中运行训练任务。

文件架构：
1. parse_args()：命令行参数解析函数
   - 定义了一个内嵌的 str2bool 类型转换器，用于将字符串参数转换为布尔值
   - 使用 argparse.ArgumentParser 定义和解析所有命令行参数
   - 参数分为四个主要类别：
     a) 通用设置（General Settings）：启动模式、调试模式
     b) SLURM 集群配置（Slurm Settings）：节点数、GPU 配置、内存、时间等
     c) Accelerate 框架配置（Accelerate Settings）：混合精度训练
     d) 训练配置（Additional Training Settings）：配置文件路径、额外选项

2. main()：主入口函数
   - 解析命令行参数
   - 根据指定的启动模式 (args.mode) 动态调用相应的启动函数
   - 支持的启动模式通过 launch_utils 模块提供：{mode}_launch (如 submitit_launch、accelerate_launch、python_launch)

3. 程序入口 (if __name__ == "__main__")：调用 main 函数启动程序

实现的功能：
-   **多模式启动支持**：支持三种启动模式
    * submitit_launch：提交任务到 SLURM 集群
    * accelerate_launch：使用 Hugging Face Accelerate 库进行分布式训练
    * python_launch：直接使用 Python 运行

-   **SLURM 集群配置**：提供完整的 SLURM 参数配置，包括：
    * 计算资源配置：节点数、每节点GPU数、每任务CPU数、内存分配
    * 作业参数：作业名称、输出目录、QoS等级、分区、账户
    * 分布式训练配置：通信端口、节点列表

-   **分布式训练支持**：通过参数配置支持多节点多GPU的分布式训练

-   **灵活的参数传递**：支持通过命令行 opts 参数传递额外的配置覆盖

-   **Accelerate 框架集成**：支持混合精度训练（no、fp16、bf16）
"""
import argparse

import common.launch_utils as lu


def parse_args():
    """
    解析命令行参数。
    
    定义并返回程序运行所需的所有命令行参数，包括启动模式、SLURM配置、
    Accelerate配置等。
    
    Returns:
        argparse.Namespace: 包含所有解析的命令行参数
    """
    def str2bool(v):
        """将字符串转换为布尔值。支持 yes/true/t/y/1 转换为 True，no/false/f/n/0 转换为 False。"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Unsupported value encountered')
    parser = argparse.ArgumentParser()

    # ============ 通用设置 (General Settings) ============
    # 启动模式：选择如何执行训练脚本（SLURM集群、Accelerate分布式、本地Python）
    parser.add_argument("--mode", default="submitit", type=str,
                        help="Launch mode (submitit | accelerate | python)")
    # 调试模式：启用时减少日志冗长性和运行时间用于快速测试
    parser.add_argument("--debug", default=False, type=str2bool,
                        help="Debug mode (True | False)")

    # ============ SLURM 集群配置 (Slurm Settings) ============
    # 指定SLURM作业的名称
    parser.add_argument("--name", default="masaccio", type=str,
                        help="Name of the job")
    # 指定要运行的启动脚本文件路径
    parser.add_argument("--run_file", default="run.py", type=str,
                        help="File position of launcher file")
    # 作业日志和输出文件的保存目录，%j 会被替换为作业ID
    parser.add_argument("--job_dir", default="jobs/%j", type=str,
                        help="Directory to save the job logs")
    # 分布式训练使用的节点数量
    parser.add_argument("--num_nodes", default=1, type=int, 
                        help="Number of nodes to use in SLURM")
    # 每个节点上使用的GPU数量
    parser.add_argument("--gpu_per_node", default=2, type=int,
                        help="Number of gpus to use in each node")
    # 每个GPU任务分配的CPU核心数
    parser.add_argument("--cpu_per_task", default=32, type=int,
                        help="Number of cpus to use for each gpu")
    # SLURM 队列的 QoS (Quality of Service) 等级
    parser.add_argument("--qos", default="level0", type=str,
                        help="Qos of the job")
    # SLURM 分区名称（如 gpu、cpu 等）
    parser.add_argument("--partition", default="gpu", type=str,
                        help="Partition of the job")
    # 指定使用的账户/项目名称
    parser.add_argument("--account", default="research", type=str,
                        help="Account of the job")
    # 每个GPU分配的内存大小（GB）
    parser.add_argument("--mem_per_gpu", default=80, type=int,
                        help="Memory allocated for each gpu in GB")
    # 作业的最大运行时间（小时）
    parser.add_argument("--time", default=24, type=int,
                        help="Time allocated for the job in hours")
    # 分布式训练中节点间通信的端口号
    parser.add_argument("--port", default=1234, type=int,
                        help="Default port for distributed training")
    # 指定特定的节点运行作业（留空表示自动分配）
    parser.add_argument("--nodelist", default="", type=str,
                        help="Default node id for distributed training")

    # ============ Accelerate 框架配置 (Accelerate Settings) ============
    # 混合精度训练模式：可选 no（默认）、fp16（半精度）或 bf16（bfloat16精度）
    parser.add_argument("--mixed_precision", default="no", type=str,
                        help="Mixed precision training, options (no | fp16 | bf16)")

    # ============ 训练配置 (Additional Training Settings) ============
    # 训练配置文件的路径（YAML格式）
    parser.add_argument("--config", default="configs/default.yaml", type=str,
                        help="Path to the config file")
    # 通过命令行传递的额外参数，用于覆盖配置文件中的设置
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Additional options to change configureation")
    return parser.parse_args()


def main():
    """
    程序主入口函数。
    
    1. 解析命令行参数
    2. 根据指定的启动模式（submitit、accelerate、python）动态调用相应的启动函数
    3. 启动函数位于 common.launch_utils 模块中，以 {mode}_launch 命名
    """
    # 解析用户在命令行中传入的所有参数
    args = parse_args()
    
    # 动态获取对应启动模式的启动函数，并执行它
    # 例如：如果 args.mode='submitit'，则调用 lu.submitit_launch(args)
    getattr(lu, f"{args.mode}_launch")(args)
    
    # 打印启动完成的消息
    print("launched")


# ============ 程序入口 ============
if __name__ == "__main__":
    """程序的入口点，当脚本直接运行时执行 main 函数。"""
    main()