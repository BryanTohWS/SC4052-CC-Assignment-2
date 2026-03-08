# SC4052 Cloud Computing Assignment 2 - PageRank

## Setup

1. Requires Python 3.8+. Check with:
```
python --version
```

2. Create and activate a virtual environment:

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Programs

**PageRank (Part 1 & 2):**
```
python pagerank.py
```
Runs a small 3-node example comparing the Iterative vs Closed Form methods, then runs both methods on the full `web-Google_10k.txt` dataset.

**AI Web Crawler for Training - Prioritisation (Challenging Part):**
```
python crawler.py
```
Runs crawl prioritisation on a small web graph, comparing a precomputed PageRank baseline against a heuristic that ranks pages by Personalised PageRank * Authority Score. 
