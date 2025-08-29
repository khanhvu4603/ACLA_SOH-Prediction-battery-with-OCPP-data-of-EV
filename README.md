# ACLA + ANODE: Battery State of Health (SOH) Prediction

## Mô tả
Project này sử dụng mô hình ACLA (Attention-based Convolutional LSTM with Augmentation) kết hợp ANODE (Augmented Neural Ordinary Differential Equations) để dự đoán trạng thái sức khỏe (SOH) của pin xe điện từ dữ liệu OCPP (Open Charge Point Protocol).

## Cấu trúc Project

```
ACLA + ANODE/
├── B1_prepocessing.py              # Tiền xử lý dữ liệu OCPP
├── B2_update_meta_user_car.py      # Cập nhật metadata xe và người dùng
├── B3_train_acla_anode.py          # Training mô hình ACLA + ANODE
├── B4_train_acla_anode_group.py    # Training theo nhóm
├── B5_prepocessing_infer.py        # Tiền xử lý cho inference
├── B6_inference.py                  # Inference cơ bản
├── B6_inference_v2_mc.py           # Inference với MC Dropout
├── B7_plot_result.py                # Vẽ biểu đồ kết quả
├── data/                            # Dữ liệu đầu vào
├── output_data/                     # Dữ liệu đã xử lý
├── output_infer/                    # Dữ liệu cho inference
└── weight/                          # Trọng số mô hình đã train
```

## Tính năng chính

- **ACLA**: Attention mechanism + Convolutional layers + LSTM
- **ANODE**: Neural ODE với augmentation để xử lý dữ liệu chuỗi thời gian
- **MC Dropout**: Monte Carlo Dropout để ước lượng uncertainty
- **Multi-task Learning**: Dự đoán SOH chính + auxiliary tasks
- **Normalization**: Chuẩn hóa dữ liệu per-feature

## Yêu cầu hệ thống

```bash
pip install torch numpy pandas torchdiffeq matplotlib scikit-learn
```

## Cách sử dụng

### 1. Training
```bash
python B3_train_acla_anode.py
```

### 2. Inference
```bash
python B6_inference_v2_mc.py
```

### 3. Vẽ kết quả
```bash
python B7_plot_result.py
```

## Dữ liệu đầu vào

- **OCPP Data**: Dữ liệu từ trạm sạc xe điện
- **Car Metadata**: Thông tin về loại xe, model
- **User Tags**: Thông tin người dùng

## Kết quả

- **SOH Prediction**: Dự đoán trạng thái sức khỏe pin
- **Uncertainty**: Độ tin cậy của dự đoán (MC Dropout)
- **Metrics**: RMSE, MAE, R² score

## Tác giả

Project được phát triển cho việc nghiên cứu và ứng dụng thực tế trong lĩnh vực xe điện.

## License

[MIT License](LICENSE)
