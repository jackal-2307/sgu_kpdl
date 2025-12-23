# TỐI ƯU HÓA CHIẾC LƯỢC TIẾP THỊ THẺ TÍN DỤNG THÔNG QUA PHÂN KHÚC KHÁCH HÀNG DỰA TRÊN HÀNH VI

**Môn:** Khai phá dữ liệu  
**Lớp:** DDU1231

**Giảng viên hướng dẫn:** TS.Đỗ Như Tài

## Phát biểu bài toán

Ngân hàng thu thập rất nhiều dữ liệu giao dịch thẻ tín dụng, nhưng việc phân khúc khách hàng vẫn thường dựa trên kinh nghiệm hoặc một vài tiêu chí đơn giản (tuổi, thu nhập, khu vực…). Cách làm này khó phản ánh đầy đủ hành vi chi tiêu, thói quen thanh toán và mức độ sử dụng hạn mức của từng khách hàng, dẫn đến:

- Chiến dịch marketing dàn trải, hiệu quả thấp  
- Khách hàng giá trị cao chưa được chăm sóc đúng mức  
- Khách hàng tiềm ẩn rủi ro tín dụng không được phát hiện sớm  

Dự án hướng tới xây dựng một cách tiếp cận **dựa trên dữ liệu hành vi**, dùng mô hình phân cụm để tự động nhóm khách hàng thành các phân khúc có ý nghĩa và dễ diễn giải.

---

## Mục tiêu

- Xây dựng pipeline phân khúc khách hàng thẻ tín dụng bằng **thuật toán K-Means** (unsupervised learning).  
- Làm sạch, tiền xử lý và chuẩn hóa bộ dữ liệu **CC_GENERAL.csv**.  
- Lựa chọn số cụm phù hợp và huấn luyện mô hình với **K = 4**.  
- Diễn giải đặc trưng của từng cụm và đề xuất một số gợi ý chiến lược cho Marketing, CSKH và Quản trị rủi ro.  

---

## Câu hỏi nghiên cứu

Trên cơ sở mục tiêu trên, nghiên cứu tập trung trả lời các câu hỏi:

1. Làm thế nào để phân loại khách hàng dựa trên thói quen sử dụng thẻ (credit usage) và hành vi thanh toán (payment behavior), nhằm hỗ trợ tối ưu hóa các chiến dịch marketing?
2. Đặc trưng hành vi nổi bật của từng cụm khách hàng là gì, và với mỗi cụm, chiến lược hành động phù hợp nhất đối với ngân hàng là gì?
3. Có thể đề xuất các định hướng chuyển dịch khách hàng từ những nhóm có giá trị hiện tại thấp sang các nhóm khách hàng có giá trị cao hơn hay không, qua đó gia tăng doanh thu và mức độ gắn bó trong dài hạn?

---

## Tập dữ liệu

- **Nguồn:** Bộ dữ liệu công khai `CC_GENERAL.csv`  
  Kaggle: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
- **Số lượng quan sát:** 8.950 khách hàng thẻ tín dụng đang hoạt động
- **Số biến đặc trưng:** 18 biến số, tập trung vào:
  - Số dư và biến động số dư (BALANCE, BALANCE_FREQUENCY)
  - Giá trị và tần suất mua sắm (PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES, …)
  - Giá trị và tần suất ứng tiền mặt (CASH_ADVANCE, CASH_ADVANCE_FREQUENCY)
  - Hạn mức tín dụng (CREDIT_LIMIT)
  - Khoản thanh toán, thanh toán tối thiểu, tỷ lệ thanh toán (PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT)
  - Thời gian gắn bó với ngân hàng (TENURE)
- **Cấp độ dữ liệu:** Mỗi dòng tương ứng với **một khách hàng**, được tổng hợp trong khoảng **6 tháng gần nhất**.

---
## sơ đồ quy trình của dự án

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/0959c324-3b26-4e6b-952b-67c1cabb0acf" />

---

## Kết quả tóm tắt

Mô hình K-Means với **4 cụm** cho thấy sự khác biệt rõ rệt giữa các nhóm khách hàng về:

- Mức độ chi tiêu và tần suất sử dụng thẻ  
- Hành vi ứng tiền mặt và thanh toán dư nợ  
- Mức độ tiềm ẩn rủi ro tín dụng  

Từ đó, có thể gợi ý các hướng chiến lược như: ưu đãi giữ chân nhóm chi tiêu cao nhưng rủi ro thấp, chương trình kích hoạt lại cho nhóm ít hoạt động, và các biện pháp kiểm soát rủi ro cho nhóm sử dụng tín dụng căng thẳng.

---

**Thành viên thực hiện**
- Phạm Hoàng Tiến -	3123580051
- Thạch Ngọc Thảo	- 3123580046
- Nguyễn Thái Tú - 3123580058
