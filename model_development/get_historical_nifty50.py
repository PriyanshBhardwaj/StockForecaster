import requests
import json
import time
import pandas as pd
import re
import datetime

url = 'https://www.google.com/finance/_/GoogleFinanceUi/data/batchexecute?rpcids=AiCwsd&source-path=%2Ffinance%2Fquote%2FNIFTY_50%3AINDEXNSE&f.sid=-984345042002505590&bl=boq_finance-ui_20240610.00_p0&hl=en&soc-app=162&soc-platform=1&soc-device=1&_reqid=5447393&rt=c'


payload = 'f.req=%5B%5B%5B%22AiCwsd%22%2C%22%5B%5B%5Bnull%2C%5B%5C%22NIFTY_50%5C%22%2C%5C%22INDEXNSE%5C%22%5D%5D%5D%2C1%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C0%5D%22%2Cnull%2C%22generic%22%5D%5D%5D&at=ANXCC_D6-Be4Ld5-EizOuzN97uhM%3A1721029189611&'


header = {
    'Accept':'application/json',
    'Accept-Encoding': 'br',
    'Accept-Language': 'en-GB,en;q=0.5',
    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
    'Cookie': 'HSID=AuuzxjtEIM3XWkP6g; SSID=Apcq6-Kj1-H3EzcfD; APISID=2HSCwa_HaBeecL8W/AavgToZJ9WnfyMBCC; SAPISID=awX93Uc7sKOPCBgb/AswOOuEi3tArHrvJ8; __Secure-1PAPISID=awX93Uc7sKOPCBgb/AswOOuEi3tArHrvJ8; __Secure-3PAPISID=awX93Uc7sKOPCBgb/AswOOuEi3tArHrvJ8; SEARCH_SAMESITE=CgQIhpsB; SOCS=CAISNQgQEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjQwNTE0LjA2X3AwGgJmaSADGgYIgOu0sgY; receive-cookie-deprecation=1; S=billing-ui-v3=3FsHfZ44WgUWWPo3ZxXDkDVC5VPCXC8i:billing-ui-v3-efe=3FsHfZ44WgUWWPo3ZxXDkDVC5VPCXC8i; SID=g.a000lghVvk2yaWJrJw_ECg2PzWj5N4xW-iH-lpjTkaZRddr8hArvJqdH_su_AvdMRCXbptHQZwACgYKAccSARcSFQHGX2MitS4OTyuc1Sfq2ygE6Na9zxoVAUF8yKoJwnXfJZlCiF7FjLNk69Gk0076; __Secure-1PSID=g.a000lghVvk2yaWJrJw_ECg2PzWj5N4xW-iH-lpjTkaZRddr8hArvCDfgq0bhG_20fj29QSQhbAACgYKARoSARcSFQHGX2MiH1rTOUh4_FqlNKHoPgRmQBoVAUF8yKq3-qMlynaVsr_aXIoKz20P0076; __Secure-3PSID=g.a000lghVvk2yaWJrJw_ECg2PzWj5N4xW-iH-lpjTkaZRddr8hArvHdAdprnqHfpE0fV1cNTI1QACgYKAf8SARcSFQHGX2Mi4ne0tRLv_51felk5Tm8I5hoVAUF8yKovO4RVe4wdemvnv-bXbEw20076; AEC=AVYB7cpTJ3WCpTuKox8vACNbtipyTytALFyff1v1yYHT-Z2mFXF_kiZmUKM; OTZ=7645327_34_34__34_; __Secure-1PSIDTS=sidts-CjIB4E2dkf8e2ohXGVvenRLPF4Pg8efbGNbg68OyKm4li1IwUvwnvp6FaERH9vfJnZWW_BAA; __Secure-3PSIDTS=sidts-CjIB4E2dkf8e2ohXGVvenRLPF4Pg8efbGNbg68OyKm4li1IwUvwnvp6FaERH9vfJnZWW_BAA; NID=515=nU8fiJxluth4qCiphdkPCRbU_vcW2dsH6je7-geeIMPeIwPeMycDoyIobdblkqyu6HcI-0osRUT4NbAGNEVjzzCq7TqsmCn9SO6ogA0ph-IjY5VTNiCJK8Ligr7-9HmVwosFZIvs6c5nUTPePta3o-p_WVNbcF9mtj0MUotDXwDpaSNdNlcFxg8TVN9JmcgulBssfLs63HrrcB2yYOZLoJFpuWTBERsQ6H_CwUl_swt-s4WaXDlH2znra5l-re-pB63Rq-V6KLuZHj0fUrDxtIT-I8cvecF4_P230-GKiGlp486L-Cst1BN8DZI13AduAVmThmn7jKrrlpRHWD176rM78ELr69d6BpC7dYYqWHvPDxqCbEfZ7O29GsxmDsgxRrto_GxCwJBOmiKU5DFzYQh0NylK-coh66C0SbSc-d98kXK0M0GP9gtqMTU; DV=s_FlpYiyaC1T4NvUndzzyhndB5JVCxkvSfGgakooqQIAAGCqGssnH9E83QAAAOBRX6N8aZALOQAAAFXO0QKIuui4DwAAAA; SIDCC=AKEyXzW3w5IwUAvL3RpeNYYLs-cwbOVT-kcPaKrmfzgrm9TTG90HJD4paSezLf2gqIkGzqwZAZo; __Secure-1PSIDCC=AKEyXzX2nfngRug8f160sD-o-RJICX0z7ZvswHPQUXgz_--nop1nlCRsJpotg-kMeGlj_Zstd8ux; __Secure-3PSIDCC=AKEyXzWfjkXvANsFMrTSf3MInzgiezugrSZ52_OeAZoc5P6saiumHiYQAr7fQV2tDv6G9Nko8epH',
    'Origin': 'https://www.google.com',
    'Priority': 'u=1, i',
    'Referer': 'https://www.google.com/'
        }


##function to extract desired output; whole day nifty50 data
def extract_whole_day_nifty50(stock_data: str):
  # Find the JSON-like part using regex
  json_like_part = re.search(r'\n\[\["wrb\.fr".*', stock_data.text).group()

  # Strip the unwanted prefix
  json_like_part = json_like_part.strip()[len('[["wrb.fr","AiCwsd","'):-len('",null,null,null,"generic"]]')]

  # Replace escaped characters
  json_like_part = json_like_part.replace('\\"', '"')

  # Load it as JSON
  data = json.loads(json_like_part)

  # Extract the relevant data
  final_data = data[0][0][3][0][1]


  dates_list = [i[0][:5] for i in final_data]

  ## getting date

  datetime_list = []

  for date in dates_list:
    # Replace None with 0 in the list
    date = [0 if x is None else x for x in date]
    # Unpack the list into the datetime constructor
    date_obj = datetime.datetime(*date)
    datetime_list.append(date_obj)
  
  # print('\n', datetime_list, '\n')

  nifty = [i[1][0] for i in final_data]

  #Print extracted data
  # print(final_data)

  return datetime_list, nifty



## main calling function to call api & extract data
def historical_time_stock(url, payload, header):
  # calling api
  response = requests.post(url, data = payload, headers=header)

  #extracting data
  dates, nifty = extract_whole_day_nifty50(response)

  # print(len(dates), len(nifty))
  return dates, nifty



## calling main function to get data
dates, nifty = historical_time_stock(url, payload, header)
# print(dates, nifty, sep='\n')

df = pd.DataFrame()
df['Time'] = dates
df['Open'] = nifty

df['Time'] = pd.to_datetime(df['Time'], unit='ms').dt.strftime("%d-%m-%Y %H:%M:%S")

print(df)



##saving to csv
# df.to_csv('../datasets/ohlc_historical.csv', index=False)

## appending to existing ohlc_historical csv
df.to_csv('../datasets/ohlc_historical.csv', mode='a', index=False, header=False)

# #####for testing model
# df.to_csv('../model_development/testing_open.csv', index=False)

