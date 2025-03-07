
import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy(
        )  # Make a copy before using inplace=True so as to avoid modifying self.data
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(
            inplace=True)  # There can be multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        # Get data
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on='order_id')

        # Handle datetime
        ship.loc[:, 'shipping_limit_date'] = pd.to_datetime(
            ship['shipping_limit_date'])
        ship.loc[:, 'order_delivered_carrier_date'] = pd.to_datetime(
            ship['order_delivered_carrier_date'])
        ship.loc[:, 'order_delivered_customer_date'] = pd.to_datetime(
            ship['order_delivered_customer_date'])
        ship.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(
            ship['order_purchase_timestamp'])

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date) /
                np.timedelta64(24, 'h'))
            if days > 0:
                return days
            else:
                return 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp)
                / np.timedelta64(24, 'h'))
            return days

        delay = ship.groupby('seller_id')\
                    .apply(delay_to_logistic_partner)\
                    .reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id')\
                   .apply(order_wait_time)\
                   .reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')

        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        # First, get only orders that are approved
        orders_approved = self.data['orders'][[
            'order_id', 'order_approved_at'
        ]].dropna()

        # Then, create a (orders <> sellers) join table because a seller can appear multiple times in the same order
        orders_sellers = orders_approved.merge(self.data['order_items'],
                                               on='order_id')[[
                                                   'order_id', 'seller_id',
                                                   'order_approved_at'
                                               ]].drop_duplicates()
        orders_sellers["order_approved_at"] = pd.to_datetime(
            orders_sellers["order_approved_at"])

        # Compute dates
        orders_sellers["date_first_sale"] = orders_sellers["order_approved_at"]
        orders_sellers["date_last_sale"] = orders_sellers["order_approved_at"]
        df = orders_sellers.groupby('seller_id').agg({
            "date_first_sale": min,
            "date_last_sale": max
        })
        df['months_on_olist'] = round(
            (df['date_last_sale'] - df['date_first_sale']) /
            np.timedelta64(1, 'M'))
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data['order_items']

        n_orders = order_items.groupby('seller_id')['order_id']\
            .nunique()\
            .reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = order_items.groupby('seller_id', as_index=False).agg(
            {'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return self.data['order_items'][['seller_id', 'price']]\
            .groupby('seller_id')\
            .sum()\
            .rename(columns={'price': 'sales'})





        # Add these methods to the Seller class in seller.py

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """
        # Get reviews data
        reviews = self.order.get_review_score()

        # Get order_items for linking reviews to sellers
        order_items = self.data['order_items'][['order_id', 'seller_id']].drop_duplicates()

        # Merge to associate reviews with sellers
        df = order_items.merge(reviews, on='order_id')

        # Calculate aggregated review metrics per seller
        result = df.groupby('seller_id', as_index=False).agg({
            'dim_is_one_star': 'mean',  # Share of one-star reviews
            'dim_is_five_star': 'mean', # Share of five-star reviews
            'review_score': 'mean'      # Mean review score
        })

        # Rename columns for clarity
        result.columns = [
            'seller_id', 'share_of_one_stars', 'share_of_five_stars', 'review_score'
        ]

        return result

    def get_review_costs(self):
        """
        Returns a DataFrame with:
        'seller_id', 'cost_of_reviews'

        Calculates the cost of reviews based on the rating:
        1 star: 100 BRL
        2 stars: 50 BRL
        3 stars: 40 BRL
        4-5 stars: 0 BRL
        """
        # Get reviews with scores
        reviews = self.order.get_review_score()

        # Get order items to link reviews to sellers
        order_items = self.data['order_items'][['order_id', 'seller_id']].drop_duplicates()

        # Merge to get all reviews per seller
        seller_reviews = order_items.merge(reviews, on='order_id')

        # Map review scores to costs
        review_cost_map = {
            1: 100,  # 1 star: 100 BRL
            2: 50,   # 2 stars: 50 BRL
            3: 40,   # 3 stars: 40 BRL
            4: 0,    # 4 stars: 0 BRL
            5: 0     # 5 stars: 0 BRL
        }

        seller_reviews['review_cost'] = seller_reviews['review_score'].map(review_cost_map)

        # Sum up the costs for each seller
        review_costs = seller_reviews.groupby('seller_id')['review_cost'].sum().reset_index()
        review_costs.columns = ['seller_id', 'cost_of_reviews']

        return review_costs

    def get_revenues(self):
        """
        Returns a DataFrame with:
        'seller_id', 'subscription_revenue', 'sales_fee_revenue', 'revenues'

        Calculates revenues based on:
        - Subscription fee: 80 BRL per month per seller
        - Sales fee: 10% of product price (excluding freight)
        """
        # Get base data needed for calculations
        sales_data = self.get_sales()  # Contains 'seller_id' and 'sales'
        dates_data = self.get_active_dates()  # Contains 'seller_id' and 'months_on_olist'

        # Merge the data
        revenues_df = sales_data.reset_index().merge(
            dates_data.reset_index(), on='seller_id', how='inner'
        )

        # Calculate subscription revenue - 80 BRL per month per seller
        revenues_df['subscription_revenue'] = revenues_df['months_on_olist'] * 80

        # Calculate sales fee - 10% of product price (excluding freight)
        revenues_df['sales_fee_revenue'] = revenues_df['sales'] * 0.1

        # Calculate total revenue
        revenues_df['revenues'] = revenues_df['subscription_revenue'] + revenues_df['sales_fee_revenue']

        return revenues_df[['seller_id', 'subscription_revenue', 'sales_fee_revenue', 'revenues']]

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist', 'share_of_one_stars',
        'share_of_five_stars', 'review_score', 'n_orders', 'quantity',
        'quantity_per_order', 'sales', 'subscription_revenue', 'sales_fee_revenue',
        'revenues', 'cost_of_reviews', 'profits']
        """
        # Get basic training data
        training_set = \
            self.get_seller_features() \
                .merge(
                self.get_seller_delay_wait_time(), on='seller_id'
            ).merge(
                self.get_active_dates().reset_index(), on='seller_id'
            ).merge(
                self.get_quantity(), on='seller_id'
            ).merge(
                self.get_sales().reset_index(), on='seller_id'
            )

        # Add review scores if available
        review_scores = self.get_review_score()
        if review_scores is not None and not review_scores.empty:
            training_set = training_set.merge(review_scores, on='seller_id', how='left')

        # Add revenue information
        revenues = self.get_revenues()
        training_set = training_set.merge(revenues, on='seller_id', how='left')

        # Add review costs
        review_costs = self.get_review_costs()
        training_set = training_set.merge(review_costs, on='seller_id', how='left')

        # Fill NaN costs (sellers with no reviews) with 0
        training_set['cost_of_reviews'] = training_set['cost_of_reviews'].fillna(0)

        # Calculate profits
        training_set['profits'] = training_set['revenues'] - training_set['cost_of_reviews']

        return training_set
