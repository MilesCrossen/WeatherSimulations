#Watt's Next EirGrid CleanerGrid Project

This project is to be used for recommendations and research into managing the intermittency of solar and wind energy. 
historical wind speed, solar radiation and energy demand data is used to
minimise the error between the electricity production of a hypothetical grid consisting solely or almost solely of renewables,
and the grid demand. The most stable configuration is found - that is, the combination of solar
and wind energy installations that produces electricity in a pattern that most closely matches grid demand, minimising the stress on external, often less
renewable factors, such as natural gas plants, coal plants, or undersea interconnectors that may source unclean energy.

The programme reads off of weather-related csv files showing patterns of solar radiation, and/or wind speed^2. These are sourced from Met Éireann's archive.
A large sample of data (20+ years) from a single weather station can be used. This data is then used to simulate
production patterns, using the knowledge that solar energy production is proportional to solar radiation, and wind energy production is proportional to wind speed^2.
Electricity demand data is sourced from EirGrid.

This data is graphed and decomposed into important components, including the fourier transform of the statistic being measured and the fourier transform of its standard deviation, which
is used for real-world simulations. Fourier transforms are used as they provide a means to describe patterns mathematically. Once fourier transforms describing wind production patterns and 
solar production patterns are found, coefficients for each of these values are found that minimise the difference between their sum and electricity demand. Absolute error minimisation is used.
This effectively finds a configuration that produces the most stable combination of the main renewable energy sources, providing a framework to adapt to challenges presented by the intermittent nature 
of these common renewables. For this part, only equations describing averages are used, and standard deviation data is used later on when real-world simulations are performed.

Various important data points are found, such as the capacity and ratio of wind and solar resources to be installed in an optimal scenario. Absolute error between their average production trends and 
energy demand is found, and five real-world simulations are run, which use the fourier transforms of wind and solar, as well as the transforms of their standard deviations and the fourier transform of demand.

The FourierAnalysis.py script can find the fourier transform of a timestamped column in a csv file. In this case, the timestamped column is either solar radiation data, or wind speed^2 data.
Wind speed^2 data is found by squaring wind speeds found in Met Éireann's historical data, inside Excel. The fourier transform of the data itself and its standard deviation (because each day of the year
has multiple data points from multiple different years) is obtained, and printed in equation form. This can be manually inputted into the SolarSimulations.py and/or WindSimulations.py, providing a mathematical basis
on which the rest of the programme operates.

Fourier analysis for demand data is found by running the DemandDataProcessing.py script in a similar fashion, except no data on standard deviation is obtained at this point because I only use
a single year of data (22-23). The equation produces is used inside DemandSimulations.py.

When main.py is run, the following graphs are produced as of 18/01/2025:
Daily electricity demand fourier transform graph,
Fourier transform of average + standard deviation of wind speed^2 graph,
Simulated daily wind production graph (5 simulations accounting for deviation data),
Fourier transform of average + standard deviation of solar energy^2 graph,
Simulated daily solar production graph (5 simulations accounting for deviation data),
Graph of sum of electricity demand using optimal coefficients for solar and wind production vs actual demand (ideally a heavy overlap should be observed),
Comparison of running sums of 5 scaled simulations (user can scale electricity production by a certain amount if they wish) vs electricity demand,
Cumulative energy surplus/deficit of 5 scaled simulations.

The following data is printed:
Optimised coefficients for wind/solar production that minimise difference between them and demand,
Total absolute error,
Average wind production in MW,
Average solar production in MW,
Ratio of recommended wind vs solar installations,
Maximum cumulative surplus/deficit of each simulation in MWh i.e. total energy that must be imported/exported under worst case scenarios.

Additionally, if the user chooses to run the PowerLimits.py script, the derivatives of the cumulative surplus/deficit of each simulation graph is used to calculate the maximum/minimum rate
of change for each simulation. This represents the maximum rate at which extra energy must be added or excess energy removed over the course of the year, for each simulation. 
Crucially, this provides information on the average amount of electricity (in MW) that must be sourced from alternative sources, such as interconnectors, coal or natural gas, during a worst-case scenario.
However, this should be improved at a later date by using simulations with a much smaller resolution, such as 15-minute or one-hour intervals. This is because using a resolution of a single day as
we do, provides data on the average energy that must be added/removed over the course of a full day in the worst case scenario, but real-world data might fluctuate very quickly and intensely within smaller
portions of a single day.

As of 29/01/2025, a separate script has been created: 
GradientDescentOptimisation.py. This file requires as inputs every
fourier transform equation for wind speed^3 or global radiation for every
weather station. With these inputs, along with the demand equation, gradient descent
is used to find optimal coefficients to multiply the solar/wind^3 equations by, to minimise
the difference between their sum and demand.

These coefficients represent capacities for solar and wind energy
installations at the corresponding weather station. We can therefore use
a large amount of solar and wind equations, from a large amount of weather stations,
to create a model for a grid that is as stable as possible while using renewables.

Shortly, I will implement actual capacity figure calculations at each of these locations.

Inside the gradient descent calculation, I have added a penalty term which disproportionately
penalises locations with low solar/wind capacity. In practicality, implementing
solar/wind installations in low-capacity locations would be uneconomical. This
figure can be adjusted (lambda_penalty), but its effect is not easily quantifiable as a
percentage weight or something else, due to the nature of how gradient descent
operates. Higher values produce harsher penalisation but cause the programme
to require more iterations to converge to an absolute error value. The iteration 
count can be adjusted using the iterations variable in 
GradientDescentOptimisation.py. In the end, similar absolute error values are generally achieved
using higher lambdas as opposed to lower lambdas,
provided enough iterations. This penalisation-based approach leads to generally
lower proportions of cost-inefficient installation types, but this can be
difficult to predict due to the volatile and numerical nature of gradient
descent. Some areas of low capacity will see increased coefficients, but in general
there is a large decrease.

An optimisation of regular gradient descent, known as adaptive movement estimation
(ADAM) is used in this. ADAM involves adapting the step size based on factors
such as previous step size and gradient. It is explained here:
https://www.geeksforgeeks.org/adam-optimizer/

Future goals as of 29/01/2025:
Restrict coefficients to non-negative values as they can currently assume negative values.
Begin performing simulations using standard deviation values, for optimal setups w/optimal coefficients.

