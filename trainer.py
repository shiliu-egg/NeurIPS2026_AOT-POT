import yaml
import subprocess
import time
import argparse
import os
import sys


class Trainer:
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
        base_args = [python_executable, self.config['file']]

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
        base_args.extend(['--log_path', log_path, '--gpu', str(gpu)])
        process = subprocess.Popen(base_args)
        print(f"Task started on GPU {gpu}")
        self.processes[gpu] = process

    def monitor_tasks(self, tasks):
        while True:
            for gpu, process in list(self.processes.items()):
                ret_code = process.poll()
                if ret_code is not None:
                    print(f"Task on GPU {gpu} has finished.")
                    del self.processes[gpu]
                    return gpu

            if tasks:
                available_gpus = self.get_available_gpus()
                free_gpus = [gpu for gpu in available_gpus if gpu not in self.processes.keys()]
                if free_gpus:
                    return free_gpus[0]

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

        available_gpus = self.get_available_gpus()

        while tasks and available_gpus:
            task = tasks.pop(0)
            gpu = available_gpus.pop(0)
            self.run_task(task, gpu)
            time.sleep(20)

        while tasks:
            free_gpu = self.monitor_tasks(tasks)
            if free_gpu is not None:
                task = tasks.pop(0)
                self.run_task(task, free_gpu)
                time.sleep(20)

        print("All tasks are either running or queued. Waiting for them to complete...")
        while self.processes:
            self.monitor_tasks([])

        print("All tasks have finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training or pretraining Neural Operators')
    parser.add_argument('--config_file',type=str,default='ns2d_pretrain.yaml')
    args = parser.parse_args()

    trainer = Trainer(os.path.join('configs',args.config_file))
    trainer.start()
