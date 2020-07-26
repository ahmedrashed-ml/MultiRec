# MultiRec-RecSys2020


This is our implementation for the recsys 2020 paper:

Ahmed Rashed, Shayan Jawed, Lars Schmidt-Thieme, and Andre Hintsches. 2020. MultiRec: A Multi-Relational Approach for UniqueItem Recommendation in Auction Systems. InFourteenth ACM Conference on Recommender Systems (RecSys ’20)

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
