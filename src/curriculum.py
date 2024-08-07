
import math

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
        print(f'lr before at step {self.step_count}: {self.learning_rate}')
        self.learning_rate = self.update_var(
            self.learning_rate, self.learning_rate_schedule
        )
        print(f'lr after at step {self.step_count}: {self.learning_rate}')
        self.sparsity_penalty = self.update_var(self.sparsity_penalty, self.sparsity_penalty_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule['interval'] == 0:
            var += schedule['increment']

        return min(var, schedule['end'])


