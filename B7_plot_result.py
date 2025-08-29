import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Đọc kết quả inference
df = pd.read_csv("soh_predictions_mc.csv")

# Lấy cột
x = df["soh_true"].values.astype(float)            # SOH true (proxy)
y = df["soh_pred_mean"].values.astype(float)       # SOH predicted

# Loại bỏ NaN (nếu có)
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]; y = y[mask]

# Hồi quy tuyến tính: y = a*x + b
a, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_fit  = a * x_line + b

# Chỉ số đánh giá
r2  = r2_score(x, y)  # R² giữa true (x) và pred (y)
rmse = np.sqrt(mean_squared_error(x, y))
mae  = mean_absolute_error(x, y)

# Vẽ
plt.figure(figsize=(7,7))
plt.scatter(x, y, alpha=0.5, label="Predictions")
plt.plot(x_line, y_fit, linewidth=2, label=f"Fit: y={a:.3f}x+{b:.3f}")
plt.plot([x.min(), x.max()], [x.min(), x.max()], linestyle="--", label="Ideal y=x")

plt.xlabel("SOH True (proxy)")
plt.ylabel("SOH Predicted")
plt.title("SOH_pred vs SOH_true (Regression + Metrics)")
plt.legend()
plt.grid(True)

# In R², RMSE, MAE lên góc trên (trong trục)
text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
# toạ độ tương đối axes (0–1): (0.02, 0.98) = góc trên trái; đổi ha='right', x=0.98 nếu muốn góc phải
plt.gca().text(0.02, 0.98, text, transform=plt.gca().transAxes,
               va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()
