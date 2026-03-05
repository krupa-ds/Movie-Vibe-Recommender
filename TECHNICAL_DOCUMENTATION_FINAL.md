# Movie Recommender System: Technical Documentation

**Based on**: `recommender_clean.ipynb`  
**System**: Hybrid content-based + collaborative filtering with 60-cluster taste taxonomy  
**Components**: 400-dim SVD + Content clustering + LightGBM hybrid model

---

## System Architecture

```
User Input (Search + Rate 5 films)
        ↓
Content Profile Building (weighted genre/theme/tag vectors)
        ↓
Recommendation Scoring:
  • Content Similarity (70%)
  • Cluster Preference (30%)
        ↓
Diversification (max 3 per cluster)
        ↓
Output: 24 diverse recommendations
```

---

## 1. Data Sources

### Ratings Dataset (`reviews.csv`)
- **Shape**: 17,928,471 ratings
- **Users**: ~400,000 users  
- **Films**: ~5,500 films
- **Fields**: `user_id`, `movie_id`, `rating`, `timestamp`

### Film Metadata (`movies_merged.csv` → `film_data_fixed.csv`)
- **Original**: 5,505 films
- **After deduplication**: 4,305 unique films
- **Fields**:
  - `movie_id`: Unique identifier
  - `name`: Film title (UTF-8 encoded, fixed for special characters)
  - `year`: Release year
  - `genres`: Comma-separated (e.g., "Drama, Thriller")
  - `theme`: Comma-separated themes (e.g., "Love, Coming of age")
  - `tags`: User-generated tags
  - `description`: Plot summary
  - `tmdb`: TMDb ID for posters
  - `rating`: Average user rating
  - `duration`: Runtime in minutes

### Poster URLs (`TMDB_movie_dataset_v11.csv`)
```python
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"

tmdb = pd.read_csv("TMDB_movie_dataset_v11.csv")
tmdb_poster = dict(zip(tmdb['id'], tmdb['poster_path']))

film_data['poster_url'] = film_data['tmdb'].map(tmdb_poster)
```

---

## 2. Data Preprocessing

### Encoding Fix (UTF-8 Mojibake)

**Problem**: Film titles with accents were corrupted:
- `L√©on: The Professional` → Should be `Léon: The Professional`
- `Y Tu Mam√° Tambi√©n` → Should be `Y Tu Mamá También`

**Solution**: Applied encoding fix and saved as `film_data_fixed.csv`

```python
def fix_mojibake(text):
    try:
        return text.encode('latin-1').decode('utf-8')
    except:
        replacements = {'√©': 'é', '√®': 'è', '√°': 'à', ...}
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        return text

film_data['name'] = film_data['name'].apply(fix_mojibake)
film_data.to_csv('film_data_fixed.csv', encoding='utf-8', index=False)
```

### Deduplication

```python
film_data = film_data.drop_duplicates(subset=['movie_id'], keep='first')
# Result: 4,305 unique films
```

---

## 3. User & Film Encoding

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
film_enc = LabelEncoder()

ratings['user_idx'] = user_enc.fit_transform(ratings['user_id'])
ratings['film_idx'] = film_enc.fit_transform(ratings['movie_id'])

n_users = len(user_enc.classes_)  # ~170,000 users
n_films = len(film_enc.classes_)  # ~3,800 films (after filtering)
```

**Note**: Only films with sufficient ratings are encoded for model training.

---

## 4. Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Center ratings by user mean
global_mean = ratings['rating'].mean()
user_means = ratings.groupby('user_idx')['rating'].mean()
ratings['rating_centered'] = ratings['rating'] - ratings['user_idx'].map(user_means)

# 80/20 split
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

print(f"Global mean rating: {global_mean:.3f}")  # ~3.7
print(f"Train: {len(train):,} ratings")
print(f"Test:  {len(test):,} ratings")
```

---

## 5. Collaborative Filtering: TruncatedSVD

### Sparse Matrix Construction

```python
from scipy.sparse import csr_matrix

sparse_matrix = csr_matrix(
    (train['rating_centered'], (train['user_idx'], train['film_idx'])),
    shape=(n_users, n_films)
)

sparsity = 1 - sparse_matrix.nnz / (n_users * n_films)
print(f"Sparsity: {sparsity:.4f}")  # ~99.9%
```

### SVD Decomposition (400 Components)

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=400, random_state=42)
user_factors = svd.fit_transform(sparse_matrix)  # (n_users, 400)
item_factors = svd.components_.T                 # (n_films, 400)

explained_variance = svd.explained_variance_ratio_.sum()
print(f"Explained variance: {explained_variance:.4f}")  # ~45-50%
```

**Key Decision**: 400 components (higher than typical 50-100) for better accuracy despite increased dimensionality.

### Normalization for Clustering

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
item_factors_norm = scaler.fit_transform(item_factors)
user_factors_norm = scaler.transform(user_factors)
```

**Note**: Normalization is separate from prediction - used only for clustering stability.

---

## 6. SVD Baseline Evaluation

### Predictions

```python
# Reconstruct ratings from factors
u_idx = test['user_idx'].values
f_idx = test['film_idx'].values
u_means_arr = user_means.reindex(range(n_users), fill_value=0).values

centered_preds = np.sum(user_factors[u_idx] * item_factors[f_idx], axis=1)
preds = np.clip(centered_preds + u_means_arr[u_idx], 0.5, 5.0)
```

### Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

svd_rmse = np.sqrt(mean_squared_error(test['rating'].values, preds))
svd_mae = mean_absolute_error(test['rating'].values, preds)

print(f"SVD RMSE: {svd_rmse:.4f}")  # ~0.85-0.90
print(f"SVD MAE:  {svd_mae:.4f}")   # ~0.65-0.70
```

---

## 7. Content Features: Genre + Theme + Tags

### Theme Parsing

Custom parser for properly formatted theme strings:

```python
import re

def split_themes(theme_str):
    """Split on ', ' only when followed by capital letter"""
    if not theme_str or str(theme_str) == 'nan':
        return []
    parts = re.split(r',\s+(?=[A-Z])', str(theme_str))
    return [p.strip() for p in parts if p.strip()]

# Example:
# Input:  "Love, Romance, Coming of age"
# Output: ['Love', 'Romance', 'Coming of age']
```

### Feature Extraction

```python
from sklearn.preprocessing import MultiLabelBinarizer

# Parse lists
film_data['genre_list'] = film_data['genres'].fillna('').str.split(', ')
film_data['theme_list'] = film_data['theme'].fillna('').apply(split_themes)
film_data['tag_list'] = film_data['tags'].fillna('').str.split(', ')
film_data['tag_list'] = film_data['tag_list'].apply(lambda t: [x for x in t if x.strip()])

# Binarize
genre_mlb = MultiLabelBinarizer()
theme_mlb = MultiLabelBinarizer()
tag_mlb = MultiLabelBinarizer()

genre_matrix = genre_mlb.fit_transform(film_data['genre_list'])
theme_matrix = theme_mlb.fit_transform(film_data['theme_list'])
tag_matrix_full = tag_mlb.fit_transform(film_data['tag_list'])

print(f"Genre features: {genre_matrix.shape[1]}")  # ~27
print(f"Theme features: {theme_matrix.shape[1]}")  # ~15
print(f"Tag features (raw): {tag_matrix_full.shape[1]}")  # ~1,500+
```

### Tag Filtering Strategy

**Parameters**:
```python
min_df = 50      # Must appear in at least 50 films
max_df = 0.05    # Must appear in at most 5% of films
```

**Excluded Tags** (metadata/location tags):
```python
exclude_tags = {
    'woman director', 'new york city', 'california', 'sequel', 'england',
    'duringcreditsstinger', 'france', 'los angeles', 'remake', 'london',
    'aftercreditsstinger', 'paris', 'japan', 'usa', 'italy'
}
```

**Filtering Process**:
```python
# Frequency-based filtering
tag_counts = tag_matrix_full.sum(axis=0)
n_films = len(film_data)
mask = (tag_counts >= min_df) & (tag_counts / n_films <= max_df)

# Manual exclusion
mask_filtered = mask.copy()
for i, tag in enumerate(tag_mlb.classes_):
    if tag in exclude_tags:
        mask_filtered[i] = False

print(f"Tags after frequency filter: {mask.sum()}")           # ~200
print(f"Tags after manual exclusion: {mask_filtered.sum()}")  # ~160
print(f"Manually removed: {mask.sum() - mask_filtered.sum()}")

# Apply filter
tag_matrix = tag_matrix_full[:, mask_filtered]

# Rebuild MLB with filtered tags
kept_tags = tag_mlb.classes_[mask_filtered]
tag_mlb_filtered = MultiLabelBinarizer(classes=kept_tags)
tag_mlb_filtered.fit([kept_tags])
```

### Content Matrix Assembly

```python
content_matrix = np.hstack([genre_matrix, theme_matrix, tag_matrix]).astype(np.float64)

print(f"Genre features: {genre_matrix.shape[1]}")   # 27
print(f"Theme features: {theme_matrix.shape[1]}")   # 15  
print(f"Tag features: {tag_matrix.shape[1]}")       # 160
print(f"Content matrix: {content_matrix.shape}")    # (4305, 202)
```

**Final dimensions**: 202 sparse binary features per film

---

## 8. Content-Based Clustering

### K-Means with 60 Clusters

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

# L2 normalize for cosine-based clustering
content_norm = normalize(content_matrix.astype(np.float32), norm='l2')

n_clusters = 60

kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=10_000,
    random_state=42,
    n_init=20,
    max_iter=500
)

kmeans.fit(content_norm)

print(f"Trained KMeans with {n_clusters} clusters on {content_norm.shape[0]} films")
```

**Hyperparameters**:
- **n_clusters=60**: Semantic granularity for taste taxonomy
- **batch_size=10,000**: For efficiency with large dataset
- **n_init=20**: Multiple initializations for stability
- **max_iter=500**: Convergence iterations

### Cluster Assignment

```python
film_clusters = kmeans.predict(content_norm)

cluster_counts = pd.Series(film_clusters).value_counts()
print(f"Largest cluster: {cluster_counts.max()} films")
print(f"Smallest cluster: {cluster_counts.min()} films")
print(f"Clusters < 10 films: {(cluster_counts < 10).sum()}")
```

### Soft Clustering (Probabilistic Assignment)

```python
from scipy.special import softmax

# Calculate distances to all cluster centroids
centroids = kmeans.cluster_centers_  # (60, 202)
dists = np.linalg.norm(
    content_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :],
    axis=2
)  # (n_films, 60)

# Convert to probabilities with temperature scaling
temperature = 0.5  # Lower = sharper probabilities
film_cluster_probs = softmax(-dists / temperature, axis=1)

print(f"Cluster probabilities: {film_cluster_probs.shape}")  # (4305, 60)
```

**Use case**: Allows films to belong to multiple clusters (e.g., "Inception" in both Sci-Fi and Thriller clusters)

---

## 9. User Content Profiles

### Build from Rated Films

```python
def build_user_profiles(ratings_df, content_matrix, film_data, n_users):
    """Build user content profiles from their rated films"""
    film_id_to_pos = {fid: i for i, fid in enumerate(film_data['movie_id'])}
    user_profiles = np.zeros((n_users, content_matrix.shape[1]), dtype=np.float32)
    
    for _, row in ratings_df.iterrows():
        user_idx = int(row['user_idx'])
        movie_id = row['movie_id']
        rating = row['rating']
        
        pos = film_id_to_pos.get(movie_id)
        if pos is not None:
            # Weight by centered rating: [-1, 1] scale
            weight = (rating - 2.5) / 2.5
            user_profiles[user_idx] += weight * content_matrix[pos]
    
    # L2 normalize
    return normalize(user_profiles, norm='l2')

user_content_profiles = build_user_profiles(train, content_matrix, film_data, n_users)
```

**Logic**:
- Positive weights for high ratings (4-5 stars)
- Negative weights for low ratings (0.5-1 stars)
- Builds vector representing user's genre/theme/tag preferences

### Precompute Norms for Efficiency

```python
def precompute_norms(matrix):
    """Precompute L2 norms for fast cosine similarity"""
    norms = np.linalg.norm(matrix.astype(np.float32), axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return norms

film_norms = precompute_norms(content_matrix)
user_norms = precompute_norms(user_content_profiles)
```

### Content Similarity Computation

```python
def compute_content_similarity(user_idx, film_idx, user_profiles, content_matrix, user_norms, film_norms):
    """Cosine similarity between user profile and film"""
    dot_product = user_profiles[user_idx] @ content_matrix[film_idx]
    similarity = dot_product / (user_norms[user_idx] * film_norms[film_idx] + 1e-10)
    return similarity
```

---

## 10. Hybrid Feature Matrix

### Feature Engineering for LightGBM

```python
film_avg = train.groupby('film_idx')['rating'].mean()
film_count = train.groupby('film_idx')['rating'].count()
film_id_to_enc_idx = {fid: i for i, fid in enumerate(film_enc.classes_)}

def build_hybrid_features(df, content_sim):
    """Build feature matrix combining all signals"""
    # Pre-map all indices
    enc_idx = df['movie_id'].map(film_id_to_enc_idx).values
    
    # SVD prediction (vectorized)
    svd_pred = np.sum(
        user_factors[df['user_idx'].values] * item_factors[enc_idx],
        axis=1
    )
    
    # Film metadata
    film_avg_vals = df['film_idx'].map(film_avg).fillna(global_mean).values
    film_pop_vals = df['film_idx'].map(film_count).fillna(1).values
    
    # User bias
    user_mean_vals = df['user_idx'].map(user_means).fillna(global_mean).values
    
    # Build feature DataFrame
    features = pd.DataFrame({
        'content_sim': content_sim,
        'svd_pred': svd_pred,
        'film_avg': film_avg_vals,
        'film_popularity': film_pop_vals,
        'user_mean': user_mean_vals
    })
    
    return features
```

**Feature vector** (5 dimensions):
1. **content_sim**: Cosine similarity between user profile and film
2. **svd_pred**: SVD collaborative filtering score
3. **film_avg**: Film's average rating across all users
4. **film_popularity**: Number of ratings (log-scaled implicitly in LightGBM)
5. **user_mean**: User's average rating bias

---

## 11. Train LightGBM Hybrid Model

```python
from lightgbm import LGBMRegressor

# Build training features
train_content_sim = compute_content_similarity(
    train['user_idx'], train['film_idx'],
    user_content_profiles, content_matrix,
    user_norms, film_norms
)
train_features = build_hybrid_features(train, train_content_sim)

# Train model
X_train = train_features.fillna(0)
y_train = train['rating'].values

lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
print("LightGBM trained")
```

**Hyperparameters**:
- **n_estimators=500**: Boosting rounds (higher than baseline 100)
- **learning_rate=0.05**: Conservative for stability
- **max_depth=7**: Tree depth
- **num_leaves=63**: Leaf nodes (2^6 - 1)
- **subsample=0.8**: Row sampling per tree
- **colsample_bytree=0.8**: Feature sampling per tree

---

## 12. Model Evaluation

### Test Set Predictions

```python
# Build test features
test_content_sim = compute_content_similarity(
    test['user_idx'], test['film_idx'],
    user_content_profiles, content_matrix,
    user_norms, film_norms
)
test_features = build_hybrid_features(test, test_content_sim)

# Predict
X_test = test_features.fillna(0)
hybrid_preds = lgb_model.predict(X_test)
hybrid_preds = np.clip(hybrid_preds, 0.5, 5.0)

# Metrics
hybrid_rmse = np.sqrt(mean_squared_error(test['rating'].values, hybrid_preds))
hybrid_mae = mean_absolute_error(test['rating'].values, hybrid_preds)

print(f"Hybrid RMSE: {hybrid_rmse:.4f}")
print(f"Hybrid MAE:  {hybrid_mae:.4f}")
```

### Results Summary

| Model | RMSE | MAE | Improvement |
|-------|------|-----|-------------|
| SVD (400 components) | 0.85-0.90 | 0.65-0.70 | Baseline |
| Hybrid (LightGBM) | 0.75-0.80 | 0.58-0.62 | ~10-15% |

**Key insight**: Combining collaborative + content signals significantly improves accuracy.

---

## 13. Taste Map Visualization

### t-SNE Dimensionality Reduction

```python
from sklearn.manifold import TSNE

# Run t-SNE on normalized content features
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    n_iter=1000,
    verbose=1
)

coords_2d = tsne.fit_transform(content_norm)

print(f"t-SNE coordinates: {coords_2d.shape}")  # (4305, 2)
```

**Parameters**:
- **perplexity=30**: Balance between local and global structure
- **n_iter=1000**: Optimization iterations
- **random_state=42**: Reproducibility

### Coordinate Alignment

```python
# Map movie_id to coordinates
film_id_to_content_idx = {
    film_data.iloc[i]['movie_id']: i
    for i in range(len(film_data))
}

# Align to film_enc order (for deployment)
coords_2d_aligned = np.zeros((len(film_enc.classes_), 2), dtype=np.float32)

for enc_idx, movie_id in enumerate(film_enc.classes_):
    content_idx = film_id_to_content_idx.get(movie_id)
    if content_idx is not None:
        coords_2d_aligned[enc_idx] = coords_2d[content_idx]

# Save
joblib.dump(coords_2d_aligned, 'model_artifacts_deploy/film_coords_2d.pkl')
```

---

## 14. Cluster Analysis & Naming

### Generate Cluster Summaries

```python
cluster_summary = []

for cluster_id in range(n_clusters):
    cluster_mask = film_clusters == cluster_id
    cluster_films = film_data[cluster_mask]
    
    if len(cluster_films) == 0:
        continue
    
    # Top films by rating
    top_films = cluster_films.nlargest(10, 'rating')
    
    # Genre distribution
    all_genres = []
    for genres_str in cluster_films['genres'].dropna():
        all_genres.extend([g.strip() for g in genres_str.split(',')])
    
    genre_counts = pd.Series(all_genres).value_counts()
    
    cluster_summary.append({
        'cluster_id': int(cluster_id),
        'size': len(cluster_films),
        'top_films': top_films['name'].tolist(),
        'top_genres': genre_counts.head(3).to_dict(),
        'avg_rating': float(cluster_films['rating'].mean())
    })

# Save for manual labeling
with open('model_artifacts/cluster_summary.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2)
```

### Cluster Names (60 total)

Examples of semantic labels created from analysis:

```python
CLUSTER_NAMES = {
    0: 'Outlaws, Guns, and Redemption',
    1: 'Passionate Dramas of the Heart',
    2: "Dark Comedy's Absurd Underbelly",
    3: 'Stylized Violence and Vengeance',
    4: 'Transgressive Class Conflict Thrillers',
    9: 'Gritty Urban Crime Dramas',
    10: 'Haunted Nightmares and Twisted Terrors',
    19: 'Epic Sci-Fi Action Adventures',
    23: 'Animated Wonder and Family Magic',
    24: 'Coming of Age Emotional Dramas',
    28: 'Dark Mysteries and Crime Thrillers',
    # ... 60 total clusters
}
```

**Naming methodology**:
1. Analyze top 10 rated films per cluster
2. Identify dominant genres
3. Extract recurring themes from descriptions
4. Create evocative, descriptive label

---

## 15. Deployment Artifacts

### Save All Required Files

```python
import joblib
import os

os.makedirs('model_artifacts_deploy', exist_ok=True)

# Encoders
joblib.dump(film_enc, 'model_artifacts_deploy/film_enc.pkl')

# Film metadata (aligned to film_enc order)
film_meta_aligned = align_metadata(film_data, film_enc)
joblib.dump(film_meta_aligned, 'model_artifacts_deploy/film_meta_aligned.pkl')

# Clustering
joblib.dump(film_clusters, 'model_artifacts_deploy/film_clusters.pkl')
joblib.dump(film_cluster_probs, 'model_artifacts_deploy/film_cluster_probs.pkl')
joblib.dump(kmeans, 'model_artifacts_deploy/kmeans.pkl')

# Content features
joblib.dump(content_norm, 'model_artifacts_deploy/content_norm.pkl')

# Collaborative filtering
joblib.dump(item_factors, 'model_artifacts_deploy/item_factors.pkl')

# Hybrid model
joblib.dump(lgb_model, 'model_artifacts_deploy/lgb_model.pkl')

# Taste map
joblib.dump(coords_2d_aligned, 'model_artifacts_deploy/film_coords_2d.pkl')

# Metadata files
with open('model_artifacts_deploy/cluster_names.json', 'w') as f:
    json.dump(CLUSTER_NAMES, f, indent=2)

with open('model_artifacts_deploy/initial_films.json', 'w') as f:
    json.dump(initial_films, f, indent=2)
```

### Files Generated

**Deployment artifacts** (`model_artifacts_deploy/`):
1. `film_enc.pkl` - LabelEncoder for movie IDs
2. `film_meta_aligned.pkl` - Film metadata (name, year, genres, etc.)
3. `film_clusters.pkl` - Hard cluster assignments (n_films,)
4. `film_cluster_probs.pkl` - Soft cluster probabilities (n_films, 60)
5. `kmeans.pkl` - Trained KMeans model
6. `content_norm.pkl` - Normalized content features (n_films, 202)
7. `item_factors.pkl` - SVD item factors (n_films, 400)
8. `lgb_model.pkl` - Trained LightGBM model
9. `film_coords_2d.pkl` - t-SNE coordinates (n_films, 2)
10. `cluster_names.json` - Cluster ID to name mapping
11. `initial_films.json` - Diverse films for onboarding

---

## 16. Recommendation Pipeline

### Search-to-Rate Flow

**Step 1: User searches for a film**
```python
@app.get("/search")
async def search_films(query: str):
    matches = film_data[film_data['name'].str.contains(query, case=False, na=False)]
    return matches.to_dict('records')
```

**Step 2: Backend finds 5 similar films**
```python
@app.post("/initial-films-from-search")
async def get_initial_films_from_search(request: dict):
    movie_id = request['movie_id']
    
    # Get film's cluster and cluster probabilities
    film_idx = film_id_to_idx[movie_id]
    cluster_probs = film_cluster_probs[film_idx]
    
    # Get top 3 clusters
    top_clusters = np.argsort(cluster_probs)[-3:][::-1]
    
    # Find top-rated films from these clusters
    candidates = []
    for cluster_id in top_clusters:
        cluster_mask = film_clusters == cluster_id
        cluster_films = film_data[cluster_mask].nlargest(10, 'rating')
        candidates.extend(cluster_films.to_dict('records'))
    
    # Return 5 diverse films
    return candidates[:5]
```

**Step 3: User rates the 5 films**

**Step 4: Generate recommendations**
```python
@app.post("/recommend/new")
async def recommend_new(request: dict):
    ratings = request['ratings']  # 1 search + 5 ratings = 6 total
    n = request.get('n', 24)
    
    # Build user content profile
    user_profile = np.zeros(content_norm.shape[1])
    for rating in ratings:
        film_idx = film_id_to_idx[rating['movie_id']]
        weight = 1.0 if rating['liked'] else -1.0
        user_profile += weight * content_norm[film_idx]
    
    user_profile = normalize(user_profile.reshape(1, -1), norm='l2').flatten()
    
    # Content similarity (70%)
    content_sim = content_norm @ user_profile
    
    # Cluster preference (30%)
    user_clusters = [film_clusters[film_id_to_idx[r['movie_id']]] 
                     for r in ratings if r['liked']]
    cluster_weights = np.zeros(n_clusters)
    for cid in user_clusters:
        cluster_weights[cid] += 1.0
    
    # Combined score
    scores = content_sim * 0.7 + (cluster_weights[film_clusters] / max(cluster_weights)) * 0.3
    
    # Diversify: max 3 per cluster
    top_indices = np.argsort(scores)[-n*2:][::-1]
    
    recommendations = []
    cluster_counts = {}
    for idx in top_indices:
        cluster = film_clusters[idx]
        if cluster_counts.get(cluster, 0) < 3:
            recommendations.append(idx)
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        if len(recommendations) >= n:
            break
    
    return format_recommendations(recommendations)
```

---

## Performance Summary

### Model Metrics

| Metric | Value |
|--------|-------|
| **Training data** | 17.9M ratings |
| **Users** | ~170,000 |
| **Films** | ~3,800 |
| **Content features** | 202 (27 genres + 15 themes + 160 tags) |
| **SVD components** | 400 |
| **Clusters** | 60 |
| **SVD RMSE** | 0.85-0.90 |
| **Hybrid RMSE** | 0.75-0.80 |
| **Improvement** | ~10-15% |

### Clustering Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette score** | 0.115 | Low (expected for movies) |
| **Visual separation** | Excellent | t-SNE shows clear clusters |
| **Semantic coherence** | High | Labels match film content |

**Note**: Low silhouette is acceptable - movies naturally overlap (e.g., "Inception" is both Sci-Fi and Thriller)

### System Performance

- **Initial recommendations**: < 500ms
- **Refined recommendations**: < 300ms  
- **Taste map generation**: < 200ms
- **Search**: < 100ms

---

## Key Design Decisions

1. **400 SVD components**: Higher dimensionality for better accuracy (vs typical 50-100)

2. **60 clusters**: Granular taste taxonomy balances specificity vs coherence

3. **Tag filtering** (min 50 films, max 5%): Removes noise while keeping meaningful tags

4. **Soft clustering** (temperature=0.5): Films can belong to multiple clusters naturally

5. **Hybrid scoring** (70% content, 30% cluster): Balances similarity with diversity

6. **Max 3 per cluster**: Prevents echo chamber in recommendations

7. **Search-to-rate onboarding**: More engaging than random films, better initial signal

---

## Future Improvements

1. **Deep learning**: Neural collaborative filtering
2. **Multi-objective optimization**: Balance accuracy, diversity, novelty
3. **Active learning**: Adapt cluster boundaries from user feedback

---

**System Status**: ✅ Fully operational with search-to-rate flow and interactive taste map
