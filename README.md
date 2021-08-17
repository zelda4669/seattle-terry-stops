# Terry Stops in Seattle

The Seattle Police Department has been under increased scrutiny in the wake of 2020’s Black Lives Matter protests, particularly the Capitol Hill Occupied Protest, and as a result of this increased scrutiny, is facing budget cuts and increased pressure for reform, including calls to abolish the police entirely. I wanted to investigate if there is clear evidence of racially motivated bias within the SPD. 
Using predictive modelling, I have analyzed [publicly available data on Terry Stops](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8) to determine what elements are most important in determining the resolution of a Terry Stop. 

![Racial Demographics of Terry Stop Subjects Compared to Racial Demographics of the city of Seattle](https://github.com/zelda4669/seattle-terry-stops/blob/main/demographics%20plot.png)

By plotting the data against [census data](https://www.census.gov/quickfacts/seattlecitywashington) for the city of Seattle, we can see that black people are extremely disproportionately represented in the population of Terry Stop subjects.

After applying multiple different types of models to the data and tuning hyperparameters, I developed a model that can predict a stop's resolution with 84% accuracy. Exploration of the most important features of these models lead to the conclusion that while there is a clear racial bias in determining who gets stopped, once the stop has been made race is not a very important indicator in predicting it's resolution.
