import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * ((self.s0 / (self.s0 + iteration)) ** self.p)


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        delta_w = self.update_weights()
        self.iteration += 1
        return delta_w


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train

        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train, y_train)
        delta_w = -lr * gradient
        self.model.w += delta_w
        return delta_w


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        X_train = self.model.X_train
        y_train = self.model.y_train
        n = X_train.shape[0]
        batch_size = min(self.batch_size, n)
        batch_index = np.random.choice(n, size=batch_size, replace=False)
        Xb, yb = X_train[batch_index], y_train[batch_index]

        # 2) вычислить градиенты на батче
        gradient = self.model.compute_gradients(Xb, yb)

        # 3) обновить веса модели
        lr = self.lr_schedule.get_lr(self.iteration)
        delta_w = -lr * gradient
        self.model.w += delta_w
        return delta_w


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def update_weights(self):
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            # TODO: инициализировать хранилища при первом вызове 
            self.grad_memory = np.zeros((num_objects, num_features), dtype=float)
            self.grad_sum = np.zeros(num_features, dtype=float)

        # TODO: реализовать SAG
        batch_size = min(self.batch_size, num_objects)
        batch_index = np.random.choice(num_objects, size=batch_size, replace=False)

        for j in batch_index:
            self.grad_sum -= self.grad_memory[j]
            self.grad_memory[j] = self.model.compute_gradients(X_train[j:j+1], y_train[j:j+1])
            self.grad_sum += self.grad_memory[j]

        avg_grad = self.grad_sum / num_objects

        lr = self.lr_schedule.get_lr(self.iteration)
        delta_w = -lr * avg_grad
        self.model.w += delta_w
        return delta_w


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        # TODO: реализовать градиентный спуск с моментумом
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.velocity is None:
            self.velocity = np.zeros(num_features, dtype=float)

        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train, y_train)

        self.velocity = self.beta * self.velocity + lr * gradient

        delta_w = -self.velocity
        self.model.w += delta_w
        return delta_w


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        # TODO: реализовать Adam по формуле из ноутбука
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.m is None:
            self.m = np.zeros(num_features, dtype=float)
            self.v = np.zeros(num_features, dtype=float)

        gradient = self.model.compute_gradients(X_train, y_train)

        lr = self.lr_schedule.get_lr(self.iteration)
        k = self.iteration + 1

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient * gradient)

        m_hat = self.m / (1.0 - self.beta1 ** k)
        v_hat = self.v / (1.0 - self.beta2 ** k)

        step = lr * m_hat / (np.sqrt(v_hat) + self.eps)
        delta_w = -step
        self.model.w = self.model.w + delta_w
        return delta_w