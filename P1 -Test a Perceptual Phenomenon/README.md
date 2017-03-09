#Project 1: Test a Perceptual Phenomenon
##Background information
In a “Stroop” task, all participants are presented with a list of words, with each word displayed in a color of ink. The participant’s task is to say out loud the color of the ink in which the word is printed. The task has two conditions: a congruent words condition, and an incongruent words condition. In the congruent words condition, the words being displayed are color words whose names match the colors in which they are printed. In the incongruent words condition, the words displayed are color words whose names do not match the colors in which they are printed. In each case, we measure the time it takes to name the ink colors in equally-sized lists. Each participant will go through and record a time from each condition.

**Dataset:** [View CSV](stroopdata.csv)

The dataset can be seen with the link above, and the remainder of this folder contains additional images etc for the project.

##Questions for investigation

####1. What is our independent variable? What is our dependent variable?
**Dependent variable:** Time to name ink colors

**Independent variable:** Condition (congruent/incongruent)

####2. What is an appropriate set of hypotheses for this task? What kind of statistical test do you expect to perform?

**Hypothesis test**

**H<sub>0</sub>:** Time to name colors is the same for congruent and incongruent tasks

**H<sub>A</sub>:** Time to name colors is *not* the same for congruent and incongruent tasks

**Statistical test**

![Normal probability plot (Congruent)](pp-congruent.png) ![Normal probability plot (Incongruent)](pp-incongruent.png)

The data is roughly normally distributed, with a slight left tail (controversial). I will use a two-tailed dependent t-test because we are comparing two dependent samples of data, which is more reliable in many instances.

####3. Report some descriptive statistics regarding this dataset.

**Congruent**
```
Mean: 14.0
SD: 3.6
```

**Incongruent**
```
Mean: 22.0
SD: 4.8
```

####4. Provide one or two visualizations that show the distribution of the sample data. Write one or two sentences noting what you observe about the plot or plots.

**Completion times of congruent and incongruent tasks**

![Completion times of congruent and incongruent tasks](completion-plot.png)

Congruent tasks appear to be consistently completed faster than incongruent tasks.

####5. What is your confidence level and your critical statistic value? Do you reject the null hypothesis or fail to reject it? Come to a conclusion in terms of the experiment task. Did the results match up with your expectations?
```
µD: -7.9648
S: 4.86482691
df: 23
t-stat: -8.020706944
at α 0.05, t-critical: -2.06865761; 2.06865761
P: 4.103E-08
95% CI: (-25.3527231, 9.42314)
```

**Null hypothesis rejected.** At α 0.05, the *time to name colors is significantly
different* between congruent and incongruent tasks. As made apparent, people do not name colors
at the same speed when the word’s meaning and its color match, as when they
do not match. The result confirms my expectations and provides proof to the experiment.

####6. Optional: What do you think is responsible for the effects observed? Can you think of an alternative or similar task that would result in a similar effect?

The brain has an image association between the shape of the word and the color. When there is a false match or mismatch, additional time is necessary for the prefrontal cortex to process the information and decide on its meaning.

A similar effect would likely be observed if the participants were shown words of the correct color but the wrong text. My thoughts on the matter, however, is that the difference would be less pronounced as I would expect the visual color representation to be more ingrained in the brain that word shape associations.

Additional Information (here for easy accessibility for the reviewer):

Since the sample subjects are the same between the Congruent and Incongruent test, the test that will be performed is a repeated measures, dependent two-condition t-test. This test is appropriate for the following reasons:
• There is a limited number of samples in each group. T-test is recommended for samples less than 30.
This data set contains only 24 observations.
• The population standard deviation is not known for either group. If it were known then the z-test
would be a more appropriate test.
• The author is assuming that both samples are approximatly normally distributed and that the variance
of the samples are not different (“Z-test and t-test,” n.d.).
The dependent variable is the time it takes to complete a test. The independent variable is the type of
test, either Congruent or Incongruent.

The point of the project is to determine if there is a statistical significance in the difference in the population mean times to perform the Congruent test (µC ) and the Incongruent test (µI ). Therefore, the null hypothesis is that there will not be a difference in the population mean times µC and µI or that µI will be less than
µC . The alternative hypothesis is that the µI will be greater than µC . These hypotheses can be expressed mathematically as:


H0 : µc ≥ µi


The t-statistic of −8.021 is far less than the t-critical value of −1.714 (see above) so the null hypothesis
is rejected. This means that there is a statistically significate difference between the population mean times
for the Congruent and Incongruent tests.

Since the alternate hypothesis is that the mean of the times for the Congruent test would be less than the
mean for the times of the Incongruent tests, this is a one-tailed t-test. The confidence interval will have a
lower bound with an upper bound of infinity.

The Cohen’s d effect size gives the difference in the two test’s mean times in terms of standard deviation.
The difference in the two test’s means is −1.637 standard deviations.
