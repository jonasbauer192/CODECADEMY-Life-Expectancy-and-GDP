from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path = "D:\\CODECADEMY-Life-Expectancy-and-GDP\\"
fileName = "all_data.csv"

class dataframe:
    def __init__(self, path, fileName):
        self.path = path
        self.filename = fileName
    def createDf(self):
        os.chdir(self.path)
        self.df = pd.read_csv(fileName)
        self.cleanDf()
    def cleanDf(self):
        # rename column
        self.df.rename(columns = {"Life expectancy at birth (years)": "Life expectancy"}, inplace=True)
        # replace column entries
        self.df['Country'] = self.df['Country'].replace('United States of America', 'USA', regex=True)

class plot:
    def __init__(self, df):
        self.df = df
        self.colors = ["brown", "red", "green", "black", "dodgerblue", "orange"]
        self.color_dict = self.createColorDict()

    def createColorDict(self):
        dct = {}
        for count, country in enumerate(list(self.df["Country"].unique())):
            dct.update({country: self.colors[count]})
        return dct

    def createMeanDf(self, column2group, column2operate):
        return self.df.groupby(column2group)[column2operate].mean().reset_index()

    def createNormalizedFrame(self, df):
        myScaler = MinMaxScaler()
        normalizedArray = myScaler.fit_transform(df[["Life expectancy", "Year"]])
        return pd.DataFrame({"Life expectancy": normalizedArray[:, 0],
                             "Year": normalizedArray[:, 1]})

    def linePlot(self):

        plotCharacteristics = [{"subplot index": 1,
                                "column": "Life expectancy",
                                "plot 1 label": "Total Mean Life Expectancy",
                                "plot 2 label": "Mean Life Expectancy of ",
                                "label y axis": "Life expectancy [Years]",
                                "title": "Life Expectancy by Year"},

                               {"subplot index": 2,
                                "column": "GDP",
                                "plot 1 label": "Total Mean GDP",
                                "plot 2 label": "Mean GDP of ",
                                "label y axis": "GDP [Dollars]",
                                "title": "GDP by Year"}]

        fig = plt.figure("Life Expectancy and GDP Trends by Year", figsize=(16, 9))
        for dct in plotCharacteristics:
            ax = plt.subplot(1, 2, dct["subplot index"])

            # plot of the mean of all countries by year
            meanDf = self.createMeanDf("Year", dct["column"])
            columns = meanDf.columns
            plt.plot(meanDf[columns[0]],
                     meanDf[columns[1]],
                     color = "black",
                     linestyle = "-",
                     marker = "o",
                     label = dct["plot 1 label"])

            # plot of all single countries by year
            for country in list(self.df["Country"].unique()):
                countryDf = self.df[self.df["Country"] == country]
                plt.plot(countryDf["Year"],
                         countryDf[dct["column"]],
                         linestyle="--",
                         marker=".",
                         label=dct["plot 2 label"] + country)

            ax.set_xticks(range(np.amin(meanDf["Year"]), np.amax(meanDf["Year"] + 4), 4))
            plt.xlabel("Year")
            plt.ylabel(dct["label y axis"])
            plt.legend(loc="best")
            plt.title(dct["title"])

        plt.savefig("Life Expectancy and GDP Trends by Year.png")
        plt.show()

    def corrleationPlot(self):

        fig = plt.figure("Corrleation between Life Expectancy and GDP", figsize=(16, 9))
        plt.subplot(3, 1, 1)
        sns.scatterplot(data=self.df, x="GDP", y="Life expectancy", hue="Country", palette=self.color_dict)
        plt.title("Life Expectancy by GDP for All Countries ")

        for counter, country in enumerate(list(self.df["Country"].unique())):
            plt.subplot(3, 3, counter + 4)
            sns.scatterplot(data=self.df[self.df["Country"] == country], x="GDP", y="Life expectancy", color=self.colors[counter])
            plt.title("Life Expectancy by GDP for " + country)

        plt.subplots_adjust(hspace=0.5)
        plt.savefig("Corrleation between Life Expectancy and GDP.png")
        plt.show()

    def distributionPlots(self):

        fig = plt.figure("Distribution of the Life Expectancy", figsize=(16, 9))
        # KDE plot
        plt.subplot(2, 2, 1)
        for country, color in zip(list(self.df["Country"].unique()), self.colors):
            countryDf = self.df[self.df["Country"] == country]
            sns.kdeplot(data=countryDf,
                        x="Life expectancy",
                        label=country)
        plt.legend(loc="upper left")
        plt.title("KDE Plot of the Life Expectancy by Country")

        # normalized KDE plot
        plt.subplot(2, 2, 3)
        for country, color in zip(list(self.df["Country"].unique()), self.colors):
            countryDf = self.df[self.df["Country"] == country]

            normalizedDf = self.createNormalizedFrame(countryDf)
            sns.kdeplot(data=normalizedDf,
                        x="Life expectancy",
                        label=country)
        plt.legend(loc="upper left")
        plt.title("KDE Plot of the Life Expectancy by Country [Normalized]")

        # boxplot
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.df, x="Country", y="Life expectancy")
        plt.xticks(rotation=45)
        plt.title("Boxplot of the Life Expectancy by Country")

        # violin plot
        plt.subplot(2, 2, 4)
        sns.violinplot(data=self.df, x="Country", y="Life expectancy")
        plt.xticks(rotation=45)
        plt.title("Violinplot of the Life Expectancy by Country")

        plt.subplots_adjust(hspace=0.5)
        plt.savefig("Distribution of the Life Expectancy.png")
        plt.show()

if __name__ == '__main__':

    # create and clean dataframe
    dfObject = dataframe(path, fileName)
    dfObject.createDf()

    # create plot object
    plotObject = plot(dfObject.df)

    plotObject.linePlot()
    plotObject.corrleationPlot()
    plotObject.distributionPlots()


