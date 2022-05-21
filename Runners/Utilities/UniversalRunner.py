import shlex, subprocess, numpy, os


class UniversalRunner:
    def __init__(self, experiment_name, param_grid, parallel=True, attempt_n=3, with_seeds=False):
        self.run(experiment_name, param_grid, parallel, attempt_n, with_seeds)
        return None

    def create(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        return None
    
    def get_param_list(self, param_grid):
        n = 1
        for key in param_grid:
            n *= len(param_grid[key])

        param_list = []
        for i in range(n):
            param_dict = {}
            k = i
            for key in param_grid:
                key_len = len(param_grid[key])
                param_dict[key] = param_grid[key][k % key_len]
                k = k // key_len
            param_list.append(param_dict)

        return param_list
    
    def run(self, experiment_name, param_grid, parallel, attempt_n, with_seeds):
        results_path = os.path.abspath('../Data')
        self.create(results_path)
        
        experiment_path = os.path.join(results_path, experiment_name)
        self.create(experiment_path)
        
        for params in self.get_param_list(param_grid):

            param_name = ''
            for key in param_grid:
                param_name += key + '=' + str(params[key]) + '_'
            param_path = os.path.join(experiment_path, param_name)
            self.create(param_path)

            for attempt in range(attempt_n):
                attempt_name = 'attempt_' + str(attempt)
                attempt_path = os.path.join(param_path, attempt_name)
                self.create(attempt_path)
                print('attempt_path' + str(attempt_path))

                cmd = experiment_name + '.py --directory "' + str(attempt_path) +'"'
                if with_seeds:
                    cmd += ' --attempt ' + str(attempt)
                for key in param_grid:
                    cmd += ' --' + key + ' ' + str(params[key])
                if parallel:
                    cmd = 'srun -n 1 -t 360 --mem-per-cpu 10000 /usr/bin/python3 ' + cmd
                else:
                    cmd = 'python ' + cmd
                print('cmd:', cmd)

                args = shlex.split(cmd)
                subprocess.Popen(args)

        return None
