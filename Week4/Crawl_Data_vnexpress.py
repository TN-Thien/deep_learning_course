import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import os
import time

# Danh sách chuyên mục
categories = {
    'https://vnexpress.net/thoi-su/chinh-tri': 'Thoi_su',
    'https://vnexpress.net/thoi-su/huong-toi-ky-nguyen-moi/tinh-gon-bo-may': 'Thoi_su',
    'https://vnexpress.net/thoi-su/chinh-tri/nhan-su': 'Thoi_su',
    'https://vnexpress.net/thoi-su/huong-toi-ky-nguyen-moi': 'Thoi_su',
    'https://vnexpress.net/thoi-su/dan-sinh': 'Thoi_su',
    'https://vnexpress.net/thoi-su/lao-dong-viec-lam': 'Thoi_su',
    'https://vnexpress.net/thoi-su/giao-thong': 'Thoi_su',
    'https://vnexpress.net/kinh-doanh/net-zero': 'Kinh_doanh',
    'https://vnexpress.net/kinh-doanh/quoc-te': 'Kinh_doanh',
    'https://vnexpress.net/kinh-doanh/doanh-nghiep': 'Kinh_doanh',
    'https://vnexpress.net/kinh-doanh/vi-mo': 'Kinh_doanh',
    'https://vnexpress.net/bong-da': 'The_thao',
    'https://vnexpress.net/the-thao/du-lieu-bong-da': 'The_thao',
    'https://vnexpress.net/the-thao/tennis': 'The_thao',
    'https://vnexpress.net/the-thao/marathon': 'The_thao',
    'https://vnexpress.net/giai-tri/gioi-sao': 'Giai_tri',
    'https://vnexpress.net/giai-tri/sach': 'Giai_tri',
    'https://vnexpress.net/giai-tri/nhac': 'Giai_tri',
    'https://vnexpress.net/giai-tri/phim': 'Giai_tri',
    'https://vnexpress.net/khoa-hoc-cong-nghe/ai': 'Cong_nghe',
    'https://vnexpress.net/khoa-hoc/bo-khoa-hoc-va-cong-nghe': 'Cong_nghe',
    'https://vnexpress.net/khoa-hoc-cong-nghe/chuyen-doi-so': 'Cong_nghe',
    'https://vnexpress.net/khoa-hoc-cong-nghe/the-gioi-tu-nhien': 'Cong_nghe',
    'https://vnexpress.net/khoa-hoc-cong-nghe/vu-tru': 'Cong_nghe',
    'https://vnexpress.net/khoa-hoc-cong-nghe/thiet-bi': 'Cong_nghe',
    'https://vnexpress.net/giao-duc/tin-tuc': 'Giao_duc',
    'https://vnexpress.net/giao-duc/tuyen-sinh': 'Giao_duc',
    'https://vnexpress.net/giao-duc/chan-dung': 'Giao_duc',
    'https://vnexpress.net/giao-duc/du-hoc': 'Giao_duc'
}

os.makedirs("Week4/data", exist_ok=True)
all_data = []

# Crawl từng chuyên mục
for url, label in categories.items():
    print(f"\nĐang crawl: {url}")
    links = set()

    for page in range(1, 11):  # Crawl 10 trang đầu
        try:
            page_url = f"{url}-p{page}"
            r = requests.get(page_url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("https://vnexpress.net") and href.endswith(".html"):
                    links.add(href.split("?")[0])
            time.sleep(1)
        except Exception as e:
            print(f"Lỗi khi tải trang: {e}")

    print(f"Tìm thấy {len(links)} bài từ {label}")

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
        except:
            continue
        time.sleep(0.5)

# Chuyển dữ liệu thu thập thành df
df = pd.DataFrame(all_data)

# Kiểm tra và loại bỏ dữ liệu trùng lặp
df["title_clean"] = df["title"].str.strip().str.lower()
duplicate_titles = df[df.duplicated(subset='title_clean', keep=False)]
df_unique = df.drop_duplicates(subset="title_clean", keep="first")

df_unique.to_csv("Week4/data/Data_vnexpress.csv", index=False, encoding="utf-8-sig") # Ghi mới dữ liệu
# df_unique.to_csv("Week4/data/Data_vnexpress.csv", index=False, encoding="utf-8-sig", mode='a', header=False) # Ghi tiếp dữ liệu
print(f"\nĐã lưu {len(df_unique)} bài vào file: Week4/data/Data_vnexpress.csv")
