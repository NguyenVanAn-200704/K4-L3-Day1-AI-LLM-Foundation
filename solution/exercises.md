# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Khi temperature tăng, câu trả lời trở nên đa dạng, sáng tạo nhưng ít dự đoán trước được hơn. Ở mức 0.0, phản hồi thường ổn định và mang tính sự thật cao; khi lên tới 1.5, câu trả lời có thể trở nên lan man hoặc thiếu chính xác về nội dung.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Với chatbot hỗ trợ khách hàng, em nên đặt temperature thấp (khoảng 0.0 đến 0.2). Điều này giúp đảm bảo câu trả lời nhất quán, chuyên nghiệp và bám sát dữ liệu thực tế, tránh việc chatbot đưa ra thông tin sai lệch do sự ngẫu hứng.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Để biết chính xác mức chênh lệch, em cần lấy giá trên mỗi 1k token của hai model từ tài liệu lab và nhân với tổng số lượng token (10.000 người _ 3 lần _ 350 token = 10,5 triệu token). GPT-4o xứng đáng khi cần tư duy logic phức tạp, viết nội dung sáng tạo cao cấp. GPT-4o-mini tối ưu cho các tác vụ phân loại nhanh, tóm tắt đơn giản hoặc chatbot hỗ trợ cơ bản để tiết kiệm chi phí vận hành lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Với system prompt là "giáo viên tiểu học", câu trả lời sẽ sử dụng từ ngữ đơn giản, câu ngắn, và dùng các ví dụ đời thường (như sổ tay ghi chép chung) để giải thích blockchain. Ngược lại, với "chuyên gia tài chính", model sẽ dùng thuật ngữ chuyên môn (như sổ cái phi tập trung, mật mã học) và cấu trúc câu phức tạp hơn. System prompt đóng vai trò là "lời đạo diễn", đặt ra ngữ cảnh và phong cách ngôn ngữ, từ đó buộc model phải thay đổi cách chọn từ và độ sâu kiến thức để phù hợp với persona được giao.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Khi đo đạc, em sẽ thấy số token từ tiktoken thường lớn hơn đáng kể so với ước lượng số từ / 0.75. Chênh lệch này thường nằm ở mức 20-50% tùy vào đoạn văn. Lý do chính là các mô hình LLM chủ yếu được huấn luyện trên dữ liệu tiếng Anh, nơi mỗi token thường tương ứng với một từ hoặc phần lớn của từ. Trong khi đó, với tiếng Việt, do đặc thù cấu tạo từ có dấu thanh và các ký tự đặc biệt, bộ mã hóa (tokenizer) thường phải tách nhỏ các từ này thành nhiều token hơn, khiến tổng số token tiêu tốn cho một câu tiếng Việt cao hơn so với câu tiếng Anh có cùng số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất trong các ứng dụng chat thời gian thực, nơi nó tạo cảm giác phản hồi tức thì bằng cách hiển thị dần dần nội dung thay vì bắt người dùng đợi toàn bộ phản hồi từ model. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý hàng loạt (batch processing), phân tích dữ liệu quy mô lớn hoặc các lệnh gọi API cần kết quả hoàn chỉnh để thực hiện các bước xử lý logic tiếp theo ngay lập tức, vì nó giúp giảm overhead do việc xử lý từng chunk nhỏ lẻ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giúp giảm tải cho server bằng cách phân tán thời gian retry của các client, tránh tình trạng hàng nghìn client cùng gửi lại yêu cầu vào một thời điểm (được gọi là "thundering herd problem"). Nếu tất cả dùng delay cố định giống nhau, khi server vừa phục hồi, nó sẽ ngay lập tức bị quá tải lại bởi một "đợt sóng" yêu cầu đồng loạt từ hàng nghìn client đó, dẫn đến khả năng cao hệ thống sẽ tiếp tục lỗi hoặc không thể hồi phục hoàn toàn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Em chọn một persona về "Trợ lý lập trình viên". System prompt: Ví dụ: "Bạn là một trợ lý lập trình viên chuyên nghiệp. Hãy trả lời ngắn gọn, tập trung vào code và giải thích bằng tiếng Việt.". Giải thích lựa chọn: Việc chỉ định "ngắn gọn" giúp tiết kiệm số lượng token đầu ra (giảm chi phí và độ trễ), trong khi chỉ định "tiếng Việt" đảm bảo tính nhất quán của phản hồi trong toàn bộ hội thoại.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> HẠN CHẾ: Trợ lý hiện tại mất toàn bộ ngữ cảnh sau khi số lượng tin nhắn vượt quá giới hạn history (bị xén), dẫn đến việc trợ lý không nhớ được các thông tin quan trọng ở phía trước. CẢI THIỆN: Triển khai cơ chế "tóm tắt lịch sử" (summarization). CÁCH TRIỂN KHAI: Thay vì chỉ cắt bỏ những tin nhắn cũ nhất, em có thể viết thêm một hàm để yêu cầu LLM tóm tắt các lượt hội thoại cũ thành một đoạn text ngắn, sau đó luôn đính kèm tóm tắt này vào system prompt để duy trì ngữ cảnh dài hạn mà không tốn quá nhiều token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
