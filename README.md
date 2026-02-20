# 🎬 Netflix Movie Recommendation System
### Collaborative Filtering on 24 Million Ratings using SVD

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 📌 Project Overview

This project builds a **personalized movie recommendation system** using the 
real-world Netflix Prize dataset. It processes over **24 million user ratings** 
across 4,499 movies and 470,000+ unique users, trains an SVD-based collaborative 
filtering model, and generates per-genre movie recommendations tailored to 
individual users.

The core idea: *given what a user has rated before, predict how much they would 
enjoy movies they haven't seen yet — and recommend the best one in every genre.*

---

## 📂 Dataset

| Source | Details |
|--------|---------|
| Netflix Prize Dataset | 24,058,263 ratings |
| Users | 470,758 unique customers |
| Movies | 4,499 titles |
| Rating Scale | 1–5 stars |
| Movie Metadata | MovieLens `movies.csv` (title + genres) |

---

## ⚙️ Project Pipeline

### 1. 🧹 Data Parsing & Cleaning
- Parsed the raw Netflix Prize `.txt` format where movie IDs are embedded 
  as row markers (e.g., `1:`) rather than a separate column
- Extracted and assigned `Movie_id` to every rating row
- Converted `Cust_id` to integer after removing movie marker rows
- Handled missing values and verified zero duplicate records

### 2. 📊 Pre-Filtering (Noise Reduction)
- Applied **quantile benchmarking** to remove low-signal data:
  - Movies with fewer than **908 ratings** (bottom 40%) removed
  - Users who rated fewer than **36 movies** (bottom 40%) removed
- Final filtered dataset: **~19.7 million ratings** used for modeling

### 3. 🤖 Model Building — SVD Collaborative Filtering
- Used **Singular Value Decomposition (SVD)** from `scikit-surprise`
- Trained on a representative sample of 1.5M ratings
- Evaluated using **3-fold cross-validation**

### 4. 🎯 Personalized Recommendations
- Generated predicted scores for all available movies for a target user
- Identified the **top-rated predicted movie per genre** (20 genres)
- Identified the **lowest-rated predicted movie per genre** (avoid list)

### 5. 📈 Genre Analysis
- Analyzed genre popularity by total rating volume
- Compared genres by average user rating (quality metric)
- Visualized best and worst predicted movies per genre

---

## 📉 Model Performance

| Metric | Score |
|--------|-------|
| RMSE (Fold 1) | 0.9755 |
| RMSE (Fold 2) | 0.9774 |
| RMSE (Fold 3) | 0.9774 |
| **Average RMSE** | **0.9768** |

> ✅ RMSE below 1.0 on a 1–5 scale is considered competitive 
> with Netflix Prize benchmarks.  
> This means our predictions are typically within **~1 star** of the true rating.

---

## 🏆 Key Findings

- **Most popular genre by volume:** Drama
- **Highest average rated genre:** Film-Noir / Documentary  
  *(niche genres attract dedicated, appreciative audiences)*
- **Top recommendation for User 2591364:** Wings of Courage (IMAX) — Score: 4.60
- **Ratings skew toward 4 stars** — consistent with voluntary rating positivity bias

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation (24M row dataset) |
| Scikit-Surprise | SVD collaborative filtering model |
| Matplotlib & Seaborn | Visualization |
| Google Colab | Runtime environment |

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Nishu2003-dev/netflix-recommendation-system

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-surprise

# 3. Add your dataset files:
#    - combined_data_1.txt (Netflix Prize)
#    - movies.csv (MovieLens metadata)

# 4. Open and run the notebook
jupyter notebook NETFLIX.ipynb
```

---

## 📁 Repository Structure
```
netflix-recommendation-system/
│
├── NETFLIX.ipynb          # Main analysis and model notebook
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── data/
    ├── combined_data_1.txt  # Netflix Prize ratings (not included — see below)
    └── movies.csv           # MovieLens movie metadata
```

> ⚠️ **Note:** The Netflix Prize dataset is not included in this repository 
> due to file size. Download it from 
> [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).

---

## 🔮 Future Improvements

- [ ] Train on the full 19.7M dataset using distributed computing
- [ ] Add content-based filtering using movie genres and metadata
- [ ] Build a hybrid recommender (collaborative + content-based)
- [ ] Deploy as a web app using Streamlit or FastAPI
- [ ] Solve cold-start problem for new users with no rating history

---

## 👤 Author

**Nishant Gupta**  
B.Tech — Computer Science with AI Specialization  
📧 nishantgupta0945@gmail.com · 💼 linkedin.com/in/nishant-gupta-98745b319 · 🐙 github.com/Nishu2003-dev

*Open to Data Science, ML Engineering, and Data Analyst roles*
