class VGTForgeController:
    def __init__(self, total_steps, warmup=4000):
        self.total_steps = total_steps
        self.warmup = warmup
        self.forge_limit = int(total_steps * 0.7)

    def get_alpha(self, step):
        # 1. Warm-up phase
        if step < self.warmup:
            return 0.0, "🔥预热"

        # 2. Forge compaction phase
        elif step < self.forge_limit:
            peak_alpha = 30.0
            progress = (step - self.warmup) / (self.forge_limit - self.warmup)
            alpha = 1.0 + (peak_alpha - 1.0) * progress
            return alpha, "⚒️压实"

        # 3. Annealing phase
        else:
            peak_alpha = 30.0
            alpha = peak_alpha * (
                1.0 - (step - self.forge_limit) / (self.total_steps - self.forge_limit)
            )
            return max(alpha, 1.0), "❄️退火"