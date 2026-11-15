# COVID Dashboard — Preview
This is a preview of the Tableau dashboard built from the SQL queries in the parent folder.
## What the dashboard shows
- Global new cases, 7-day average, Jan 2020 – Apr 2021
- Total deaths by country, Top 10: United States 576,232; Brazil 403,781; Mexico 216,907; India 211,853; United Kingdom 127,775
- Percentage of population infected, Top 10: Andorra 17.1%, Montenegro 15.5%, Czechia 15.2%, San Marino 14.9%, Slovenia 11.6%
- United States vaccination progress: about 43.8% with at least one dose by 2021-04-30
## Key numbers (to 2021-04-30)
- Total cases: 150,574,977
- Total deaths: 3,180,206
- Death rate: 2.11%
- Countries: 210
## How to reproduce in Tableau / Power BI
1. Load CovidDeaths and CovidVaccinations (Our World in Data, via AlexTheAnalyst/PortfolioProjects).
2. Run the 4 queries in tableau-covid-dashboard-queries.sql in the parent folder.
3. Build 4 visuals from those result sets: a line chart for daily cases, a bar chart for deaths by country, a bar chart for infection %, and a line chart for US vaccination progress.
4. Add the global totals above as KPI tiles.
## Data and source
Data: CovidDeaths and CovidVaccinations, 2020-01-01 to 2021-04-30. SQL queries adapted from AlexTheAnalyst/PortfolioProjects (https://github.com/AlexTheAnalyst/PortfolioProjects), adapted by Louis Lu, Nov 2026. Dashboard image: covid-dashboard.png in this folder.
