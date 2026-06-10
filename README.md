<div align="center">

# Algo Trading Bot

**Kurumsal kalitede hibrit pipeline algoritmik trading sistemi**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-managed-60A5FA?style=flat-square&logo=poetry&logoColor=white)](https://python-poetry.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## Mimari

Üç bağımsız katman ortak bir veri depolama üzerinde çalışır:

```
┌─────────────────────────────────────────────────────────┐
│                     Data Layer                           │
│              CCXT  ──────  ArcticDB                      │
└───────────┬─────────────────────────┬────────────────────┘
            │                         │
    ┌───────▼──────┐         ┌────────▼──────────┐
    │ Research     │         │ Execution          │
    │ VectorBT     │         │ NautilusTrader     │
    │ Backtesting  │         │ Live Trading       │
    └──────────────┘         └────────────────────┘
```

| Katman | Teknoloji | Görev |
|--------|-----------|-------|
| **Research** | VectorBT | Vektörleştirilmiş backtesting |
| **Data** | CCXT + ArcticDB | 100+ borsa verisi ve zaman serisi depolama |
| **Execution** | NautilusTrader | Düşük-gecikmeli canlı emir yönetimi |
| **AI/ML** | ONNX | Model inference |

---

## Özellikler

- **Çoklu borsa:** CCXT üzerinden 100+ borsa desteği
- **Hızlı backtesting:** VectorBT ile NumPy tabanlı vektörleştirilmiş hesaplama
- **Canlı trading:** NautilusTrader ile production-grade emir yönetimi
- **Zaman serisi veritabanı:** ArcticDB ile yüksek performanslı veri depolama
- **AI destekli:** ONNX model entegrasyonu
- **Docker:** Tam containerize deployment

---

## Kurulum

**Gereksinimler:** Python 3.11+, [Poetry](https://python-poetry.org/docs/#installation)

```bash
# Bağımlılıkları yükle
poetry install

# Virtual environment aktifleştir
poetry shell
```

### Docker ile

```bash
docker-compose up -d
```

---

## Kullanım

```bash
# Piyasa verisi topla
python run_ingestion.py

# Araştırma / backtesting
python run_research.py

# Strateji optimizasyonu
python run_optimization.py

# Hızlı optimizasyon
python run_optimization_fast.py
```

---

## Proje Yapısı

```
.
├── src/
│   ├── data/           # CCXT adaptörleri, ArcticDB bağlayıcıları
│   ├── analysis/       # VectorBT araştırma modülleri
│   ├── execution/      # NautilusTrader strateji sınıfları
│   ├── models/         # ONNX model yükleme ve inference
│   └── utils/          # Logger, helpers
├── tests/              # Unit ve integration testler
├── scripts/            # Yardımcı scriptler
├── notebooks/          # Jupyter araştırma defterleri
├── config/             # Konfigürasyon dosyaları
├── run_ingestion.py
├── run_research.py
├── run_optimization.py
└── pyproject.toml
```

---

## Lisans

[MIT](LICENSE) © 2026 furkntrg41
