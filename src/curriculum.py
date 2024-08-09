
class Curriculum:
    def __init__(self, args, total_iterations):
        self.learning_rate = args['learning_rate']['start']
        self.learning_rate_schedule = args['learning_rate']
        self.sparsity_penalty =  args['sparsity_penalty']['start']
        self.sparsity_penalty_schedule =  args['sparsity_penalty']
        self.total_iterations = total_iterations

        self.step_count = 0

    def update(self):
        self.step_count += 1
        
        #decrease LR linearly for last 20% of steps
        if self.step_count >= 0.8 * self.total_iterations:
            #print(f'lr before at step {self.step_count}: {self.learning_rate}') 
            decrement = self.learning_rate / (0.2 * self.total_iterations)
            self.learning_rate = self.update_var_interpolate(
                self.learning_rate, self.learning_rate_schedule, decrement, decrease=True
            )
            #print(f'lr after at step {self.step_count}: {self.learning_rate}')
        
        #increase sparsity_penalty linearly for first 5% of steps
        if self.step_count <= 0.05 * self.total_iterations:
            increment = self.sparsity_penalty_schedule['end'] / (0.05 * self.total_iterations)
            self.sparsity_penalty = self.update_var_interpolate(self.sparsity_penalty, self.sparsity_penalty_schedule, increment)

    def update_var_interval(self, var, schedule, decrease=False):
        if self.step_count % schedule['interval'] == 0:
            var += schedule['increment']

        if decrease:
            return max(var, schedule['end'])
        else:
            return min(var, schedule['end'])
    def update_var_interpolate(self, var, schedule, val, decrease=False):
        var += val
        if decrease:
            return max(var, schedule['end'])
        else:
            return min(var, schedule['end'])




