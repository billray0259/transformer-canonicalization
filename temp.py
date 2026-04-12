import unittest

class TestPinckneyStreetMath(unittest.TestCase):

    def setUp(self):
        # --- BASE ASSUMPTIONS & CASE DATA ---
        
        # Uses of Funds (Exhibit B)
        self.purchase_price = 1000000
        self.renovation = 450000
        self.carrying_costs = 36010
        self.legal_fees_own = 1500
        self.legal_fees_bank = 2000
        self.title_insurance = 4000
        self.origination_fee = 10500
        self.tax_escrow = 2441

        # Sources of Funds
        self.first_mortgage = 1050000
        self.second_mortgage = 200000
        self.lee_max_equity = 240000
        
        # Operational / Rental Data (Exhibit A & D)
        self.gross_rental_income = 157200
        self.vacancy_allowance = 7860 # 5%
        self.operating_expenses = 46846
        self.first_mtg_service = 58353
        self.second_mtg_service = 12884

        # Condo Conversion Data (Exhibit C)
        self.condo_gross_proceeds = 600000 + 650000 + 700000
        self.sales_cost_rate = 0.07
        self.penthouse_retained_value = 500000
        
        # Downside Scenario
        self.downside_floor_price = 500000

    def test_total_uses_of_funds(self):
        """Verifies the true total cost of the project against the flawed case math."""
        actual_total_uses = (
            self.purchase_price + 
            self.renovation + 
            self.carrying_costs + 
            self.legal_fees_own + 
            self.legal_fees_bank + 
            self.title_insurance + 
            self.origination_fee + 
            self.tax_escrow
        )
        # The case incorrectly sums this to $1,506,001. We test for the true sum.
        self.assertEqual(actual_total_uses, 1506451, "Total Uses math is fundamentally broken in the source document.")

    def test_capital_stack_insolvency(self):
        """Proves the structural $16,451 shortfall and the 'phantom' reserve double-count."""
        total_uses = 1506451
        total_debt = self.first_mortgage + self.second_mortgage
        cash_required_to_close = total_uses - total_debt
        
        # Lee only has $240,000 total. We calculate the deficit.
        shortfall = cash_required_to_close - self.lee_max_equity
        
        self.assertEqual(shortfall, 16451, "The $16,451 shortfall was calculated correctly.")
        
        # To have a $50k reserve, Lee would actually need:
        true_required_equity = cash_required_to_close + 50000
        self.assertEqual(true_required_equity, 306451, "Lee needs $306,451 in cash to execute the proposed plan with a $50k reserve, not $240,000.")

    def test_rental_cash_flow(self):
        """Verifies the rental hold operating metrics."""
        net_rental_income = self.gross_rental_income - self.vacancy_allowance
        noi = net_rental_income - self.operating_expenses
        
        total_debt_service = self.first_mtg_service + self.second_mtg_service
        before_tax_cash_flow = noi - total_debt_service
        
        self.assertEqual(round(noi), 102494, "NOI matches case projections.")
        self.assertEqual(round(total_debt_service), 71237, "Annual debt service calculation is correct.")
        self.assertEqual(round(before_tax_cash_flow), 31257, "BTCF of $31,257 matches projections.")

    def test_condo_conversion_returns(self):
        """Verifies the primary exit strategy yields."""
        sales_costs = self.condo_gross_proceeds * self.sales_cost_rate
        net_proceeds = self.condo_gross_proceeds - sales_costs
        
        self.assertEqual(net_proceeds, 1813500, "Net proceeds calculation is correct.")
        
        cash_to_lee = net_proceeds - self.first_mortgage - self.second_mortgage
        self.assertEqual(cash_to_lee, 563500, "Liquid cash generation is accurate.")
        
        total_wealth_created = cash_to_lee + self.penthouse_retained_value
        self.assertEqual(total_wealth_created, 1063500, "Total wealth creation math holds.")
        
        # Return on Equity = Total Wealth / Initial Equity (Assuming he actually had the cash to close)
        roe = total_wealth_created / self.lee_max_equity
        self.assertAlmostEqual(roe, 4.43125, places=4, msg="4.4x multiple is verified.")

    def test_downside_condo_scenario(self):
        """Stress-tests the downside risk at $500k per unit."""
        gross_proceeds = self.downside_floor_price * 3
        net_proceeds = gross_proceeds * (1 - self.sales_cost_rate)
        
        cash_to_lee = net_proceeds - self.first_mortgage - self.second_mortgage
        self.assertEqual(cash_to_lee, 145000, "Downside cash generation of $145,000 is perfectly accurate.")

if __name__ == '__main__':
    unittest.main(verbosity=2)