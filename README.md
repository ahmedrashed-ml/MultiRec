# MultiRec


This is our implementation for the recsys 2020 paper:

Rashed, Ahmed, et al. "MultiRec: A Multi-Relational Approach for Unique Item Recommendation in Auction Systems."14th ACM Conference on Recommender Systems (RecSys). 2020.

## Enviroment 
	* pandas==1.0.3
	* tensorflow==1.14.0
	* matplotlib==3.1.3
	* numpy==1.18.1
	* six==1.14.0
	* scikit_learn==0.23.1
  
 ## Steps
1. Download the eBay dataset ("https://www.kaggle.com/onlineauctions/online-auctions-dataset/data#auction.csv")
2. Place the auction.csv file under Data/ebay/
3. Run the data preprocessing file "python DataPrep.py"
4. To reproduce the paper results please run the following command "python MultiRec.py 42 1 1 0"

## Paper
Preprint version : https://www.ismll.uni-hildesheim.de/pub/pdfs/Ahmed_RecSys20.pdf

