import yaml
import subprocess
import time
import random
import argparse
import os
import sys


class ParallelTrainer:
    def __init__(self, yaml_path):
        self.config = self.load_yaml(yaml_path)
        self.processes = {}

    @staticmethod
    def load_yaml(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def get_available_gpus(self):
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        lines = output.split("\n")

        free_gpus = [int(line.split(",")[0].strip()) for line in lines if
                     int(line.split(",")[1].strip().split(" ")[0]) < 800]

        if 'device' in self.config:
            specified_gpus_str = self.config['device']
            specified_gpus = [int(gpu) for gpu in specified_gpus_str.split(',')]
            free_gpus = [gpu for gpu in free_gpus if gpu in specified_gpus]

        return free_gpus

    def run_task(self, task_args, gpu):
        python_executable = sys.executable
        base_args = [self.config['file']]
        for key, value in self.config.items():
            if 'task' not in key and key not in ['name', 'file', 'device']:
                if isinstance(value, bool):
                    if value:
                        base_args.append(f'--{key}')
                elif isinstance(value, list):
                    base_args.append(f'--{key}')
                    for item in value:
                        base_args.append(str(item))
                else:
                    base_args.extend([f'--{key}', str(value)])
        for key, value in task_args.items():
            if isinstance(value, list):
                base_args.append(f'--{key}')
                for item in value:
                    base_args.append(str(item))
            else:
                base_args.extend([f'--{key}', str(value)])

        log_path = f"{self.config['model']}_{self.config['dataset']}_{time.strftime('%m%d_%H_%M_%S')}"
        base_args.extend(['--log_path', log_path])
        num_gpus = task_args.get('num_gpus', self.config.get('num_gpus', 1))
        if num_gpus > 1:
            cmd = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu))} accelerate launch --num_processes={num_gpus} --multi_gpu --main_process_port 50{random.randint(10, 99)} {' '.join(base_args)}:"
        else:
            python_executable = sys.executable
            cmd = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu))} {python_executable} {' '.join(base_args)}"

        process = subprocess.Popen(cmd,shell=True)
        print(f"Task started on GPU {gpu}")
        self.processes[tuple(gpu)] = process

    def monitor_tasks(self, tasks):
        while True:
            for gpu_key, process in list(self.processes.items()):
                ret_code = process.poll()
                if ret_code is not None:
                    print(f"Task on GPUs {', '.join(map(str, gpu_key))} has finished.")
                    del self.processes[gpu_key]
                    return gpu_key

            if tasks:
                available_gpus = self.get_available_gpus()
                next_task_num_gpus = tasks[0].get('num_gpus',self.config.get('num_gpus', 1))
                if len(available_gpus) >= next_task_num_gpus:
                    return available_gpus[:next_task_num_gpus]

            time.sleep(5)
        return None

    def start(self):
        print("Starting Trainer...")

        task_params = self.config.get('tasks', {})

        if task_params:
            n = None
            for key, values in task_params.items():
                if not isinstance(values, list):
                    task_params[key] = [values]
                if n is None:
                    n = len(task_params[key])
                elif n != len(task_params[key]):
                    raise ValueError("All parameter sequences in 'tasks' should have the same length.")

            tasks = []
            num_tasks = n
            for i in range(num_tasks):
                task = {key: values[i] for key, values in task_params.items()}
                tasks.append(task)
        else:
            tasks = [{k: v for k, v in self.config.items() if k not in ['name', 'file', 'device', 'tasks']}]

        while tasks:
            available_gpus = self.get_available_gpus()
            task = tasks[0]
            num_gpus_required = task.get('num_gpus', self.config.get('num_gpus',1))

            if len(available_gpus) >= num_gpus_required:
                tasks.pop(0)
                gpu_ids_for_task = available_gpus[:num_gpus_required]
                self.run_task(task, gpu_ids_for_task)
                available_gpus = available_gpus[num_gpus_required:]
                time.sleep(20)
            else:
                break

        while tasks:
            free_gpu = self.monitor_tasks(tasks)
            if free_gpu is not None:
                task = tasks.pop(0)
                self.run_task(task, list(free_gpu))
                time.sleep(20)
        print("All tasks are either running or queued. Waiting for them to complete...")
        while self.processes:
            self.monitor_tasks([])

        print("All tasks have finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Training or pretraining Neural Operators')
    parser.add_argument('--config_file',type=str,default='pretrain_tiny.yaml')
    args = parser.parse_args()

    trainer = ParallelTrainer(os.path.join('configs',args.config_file))
    trainer.start()
