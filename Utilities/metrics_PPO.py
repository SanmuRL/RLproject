import numpy as np
import time, datetime
import matplotlib.pyplot as plt

class MetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLoss':>15}{'MeanLoss1':>15}{'MeanLoss2':>15}{'MeanValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_avg_loss_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_values_plot = save_dir / "values_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_avg_values = []
        self.ep_avg_loss = []
        self.ep_avg_loss1 = []
        self.ep_avg_loss2 = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_avg_values = []
        self.moving_avg_ep_avg_loss = []
        self.moving_avg_ep_avg_loss1 = []
        self.moving_avg_ep_avg_loss2 = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()


    def log_step(self, value):
        self.curr_ep_value += value
        self.curr_ep_length += 1

    def log_episode(self, loss, avg_rewards, l1, l2):
        "Mark end of episode"
        self.ep_rewards.append(avg_rewards)
        ep_avg_value = np.round(self.curr_ep_value / self.curr_ep_length, 5)
        self.ep_avg_values.append(ep_avg_value)
        self.ep_avg_loss.append(loss)
        self.ep_avg_loss1.append(l1)
        self.ep_avg_loss2.append(l2)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_value = 0.0
        self.curr_ep_length = 0

    def record(self, episode, epsilon):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-10:]), 3)
        mean_ep_value = np.round(np.mean(self.ep_avg_values[-10:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_loss[-10:]), 3)
        mean_ep_loss1 = np.round(np.mean(self.ep_avg_loss1[-10:]), 3)
        mean_ep_loss2 = np.round(np.mean(self.ep_avg_loss2[-10:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_avg_values.append(mean_ep_value)
        self.moving_avg_ep_avg_loss.append(mean_ep_loss)
        self.moving_avg_ep_avg_loss1.append(mean_ep_loss1)
        self.moving_avg_ep_avg_loss2.append(mean_ep_loss2)


        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Loss1 {mean_ep_loss1} - "
            f"Mean Loss2 {mean_ep_loss2} - "
            f"Mean Value {mean_ep_value} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_loss:15.3f}{mean_ep_loss1:15.3f}{mean_ep_loss2:15.3f}{mean_ep_value:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_avg_loss", "ep_avg_values"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()