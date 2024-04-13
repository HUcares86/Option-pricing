These are different option pricing methods for pricing different options.



Pricing European options:
  1.BlackScholes
  2.CRR binomial tree(1 column)
  3.CRR binomial tree
  4.Combinatorial method
  5.Monte Carlo simulation



Pricing American options:
  1.CRR binomial tree(1 column)
  2.CRR binomial tree



Pricing Rainbow options:
  Inside Rainbow option.py there are three methods
    1.Monte Carlo simulation
    2.Monte Carlo simulation with variance reduction
    3.Monte Carlo simulation with inverse Cholesky



Pricing Lookback options:
  Inside Lookback option.py there are four methods
    1.Monte Carlo simulation to price both European lookback puts
    2.binomial tree model to price both European and American lookback puts
    3.quick approach: Based on the same binomial tree framework, devise and implement a quick approach to determine the Smax list for each node to price lookback puts.
    4.Cheuk_and_Vorst : Implement the method in Cheuk and Vorst (1997) to price European and American lookback puts.


Pricing Arithmetic average options:
  Inside Arithmetic average.py there are two methods
      1.Monte Carlo simulation 
      2.binomial tree model 


Using implicit and explicit finite difference methods:
  Inside Implicit and explicit fdm.py there are two different options we can price
    1.American options
    2.European options



Using least-squares Monte Carlo simulation:
  Inside lsm.py there are three different options we can price
      1.American-style plain vanilla puts
      2.lookback puts
      3.arithmetic average calls
    with two different regression parametres (you can add your own regression parametres)
