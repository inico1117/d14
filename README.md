# d14
#regression
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Close']*100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
print(df.head()) =>             Adj. Close    HL_PCT  PCT_change  Adj. Volume
                    Date                                                     
                    2004-08-19   50.322842  3.712563    0.323915   44659000.0
                    2004-08-20   54.322689  0.710922    6.739913   22834300.0
                    2004-08-23   54.869377  3.729433   -1.243144   18256100.0
                    2004-08-24   52.597363  6.417469   -6.074187   15247300.0
                    2004-08-25   53.164113  1.886792    1.169811    9188600.0

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head()) =>             Adj. Close    HL_PCT  PCT_change  Adj. Volume      label
                    Date                                                                
                    2004-08-19   50.322842  3.712563    0.323915   44659000.0  69.078238
                    2004-08-20   54.322689  0.710922    6.739913   22834300.0  67.839414
                    2004-08-23   54.869377  3.729433   -1.243144   18256100.0  68.912727
                    2004-08-24   52.597363  6.417469   -6.074187   15247300.0  70.668146
                    2004-08-25   53.164113  1.886792    1.169811    9188600.0  71.219849
