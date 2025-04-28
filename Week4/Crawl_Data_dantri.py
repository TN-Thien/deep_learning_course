import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import os
import time

# Danh sách chuyên mục từ Dân Trí
categories = {
    'https://dantri.com.vn/xa-hoi.htm': 'Thoi_su',
    'https://dantri.com.vn/kinh-doanh.htm': 'Kinh_doanh',
    'https://dantri.com.vn/the-thao.htm': 'The_thao',
    'https://dantri.com.vn/giai-tri.htm':'Giai_tri',
    'https://dantri.com.vn/cong-nghe.htm':'Cong_nghe',
    'https://dantri.com.vn/giao-duc.htm':'Giao_duc'
}

# Tạo thư mục lưu nếu chưa có
os.makedirs("Week4/data", exist_ok=True)

all_data = []

# Crawl dữ liệu từ các chuyên mục
for url, label in categories.items():
    print(f"\nĐang crawl: {url}")
    links = set()

    # Crawl 20 trang đầu
    for page in range(1, 21):
        try:
            if page == 1:
                page_url = url
            else:
                # Thay đổi cấu trúc URL cho đúng
                page_url = f"{url.split('.htm')[0]}/trang-{page}.htm"

            r = requests.get(page_url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = "https://dantri.com.vn" + href
                if href.startswith("https://dantri.com.vn") and href.endswith(".htm"):
                    links.add(href.split("?")[0])
            time.sleep(1)
        except Exception as e:
            print(f"Lỗi khi tải trang: {e}")

    print(f"Tìm thấy {len(links)} bài từ chuyên mục {label}")

    # Crawl từng bài từ các links thu được
    for idx, link in enumerate(links):
        try:
            article = Article(link, language='vi')
            article.download()
            article.parse()
            if len(article.text.strip()) > 200:
                all_data.append({
                    "title": article.title,
                    "content": article.text,
                    "label": label
                })
        except Exception as e:
            print(f"Lỗi khi xử lý bài: {link} - {e}")
            continue
        time.sleep(0.5)

# Đưa vào DataFrame
df = pd.DataFrame(all_data)

# Loại bỏ trùng lặp theo tiêu đề
df["title_clean"] = df["title"].str.strip().str.lower()
df_unique = df.drop_duplicates(subset="title_clean", keep="first")

# Lưu vào file mới
output_path = "Week4/data/Data_dantri.csv"
df_unique = df_unique.drop(columns=["title_clean"])
df_unique.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nĐã lưu {len(df_unique)} bài vào file: {output_path}")
