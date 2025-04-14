# SAÜChat - Sakarya Üniversitesi Yönerge ve Yönetmelikler Bilgi Sistemi

## Proje Hakkında

SAÜChat, Sakarya Üniversitesi'nin resmi yönerge ve yönetmeliklerine dair sorulara yanıt veren bir bilgi sistemidir. RAG (Retrieval-Augmented Generation) mimarisi kullanarak, öğrencilerin, akademisyenlerin ve idari personelin üniversite kuralları ve prosedürleri hakkında hızlı ve doğru bilgiye erişmelerini sağlar.

## Özellikler

- PDF dosyalarından metin çıkarma ve işleme
- Metin temizleme ve anlamlı parçalara ayırma
- FAISS vektör veritabanı ile hızlı bilgi erişimi
- LLaMA 3 8B tabanlı doğal dil işleme
- Türkçe dil desteği

## Nasıl Çalışır?

1. Sistem önce kullanıcı sorusunu alır
2. Vector database içinde benzer içerikleri arar
3. Bulunan en alakalı kaynakları değerlendirir
4. LLaMA 3 8B modeli ile doğru ve tutarlı bir yanıt üretir

## Kurulum

### Gereksinimler

- Python 3.9+
- PyPDF2 veya PyMuPDF (fitz)
- NLTK
- FAISS-CPU
- SentenceTransformers
- LLaMA modelini çalıştırmak için yeterli RAM

### Adımlar

```bash
# 1. Gerekli kütüphaneleri yükleme
pip install PyPDF2 nltk faiss-cpu sentence-transformers tqdm

# 2. NLTK kaynaklarını indirme
python -c "import nltk; nltk.download('punkt')"

# 3. PDF dosyalarından metin çıkarma
python pdf_text_extract.py /path/to/pdf_folder extracted_texts

# 4. Metinleri işleme ve parçalara ayırma
python data_preprocess.py extracted_texts --output_dir processed_chunks

# 5. FAISS vektör veritabanı oluşturma
python faiss_index_process.py --input_file processed_chunks/processed_chunks.json --db_path vector_db --create_new
```

## Düşük RAM Kaynaklarında Çalıştırma

LLaMA 3 8B modelinin RAM sınırlaması olan ortamlarda çalıştırılması için öneriler:

- 8-bit quantization kullanımı
- GGUF formatında daha optimize edilmiş model kullanımı
- Gradient checkpointing
- Daha küçük bağlam penceresi ve batch size kullanımı
- Daha az belge getiren retrieval optimizasyonu

```python
# 8-bit quantization örneği
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    load_in_8bit=True,
    device_map="auto"
)
```

## Kullanım

Kullanıcı sorgularını işlemek için:

```bash
python vector_db_helpers.py --query "Staj defteri nasıl doldurulur?" --db_path vector_db
```

## Dosya Yapısı

- `pdf_text_extract.py`: PDF'lerden metin çıkarma
- `data_preprocess.py`: Metinleri temizleme ve parçalara ayırma
- `faiss_index_process.py`: FAISS vektör veritabanı oluşturma ve güncelleme
- `vector_db_helpers.py`: Sorgu işleme ve ilgili bilgileri getirme
- `extracted_texts/`: PDF'lerden çıkarılan ham metinler
- `processed_chunks/`: İşlenmiş metin parçaları
- `vector_db/`: FAISS vektör veritabanı

## Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch'i oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inize push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında lisanslanmıştır.

## İletişim

Proje sahibi: [İsim/İletişim Bilgileri]