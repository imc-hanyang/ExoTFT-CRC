import os
import pandas as pd

path = '/content/gdrive/MyDrive/reforecast'
datasets_path = path+'/dataset/solar_stations'

files = [os.path.join(path+'/dataset/solar_stations', file) for file in os.listdir(path+'/dataset/solar_stations') if file.endswith('.xlsx')]
origin_df_list = []
files.sort()
for file in files:
  print(file)
  df = pd.read_excel(file)
  print(df.columns)
  origin_df_list.append(df)


import os
import glob
import re
import numpy as np
import pandas as pd

# 1. 경로 설정
input_dir = '/content/gdrive/MyDrive/reforecast/dataset/solar_stations'
output_dir = '/content/gdrive/MyDrive/reforecast/timexer/data/origin'

os.makedirs(output_dir, exist_ok=True)

file_paths = glob.glob(os.path.join(input_dir, '*.xlsx'))

col_mapping = {
    'Global horizontal irradiance (W/m2)': 'GHI',
    'Direct normal irradiance (W/m2)': 'DNI',
    'Total solar irradiance (W/m2)': 'TSI',
    'Air temperature  (°C) ': 'Temperature',
    'Atmosphere (hpa)': 'Atmospheric pressure'
}

print(f"총 {len(file_paths)}개의 파일을 검사합니다...\n")

for file_path in file_paths:
    filename = os.path.basename(file_path)

    # 파일명에서 정격 용량(Nominal) 추출
    site_match = re.search(r'site\s*(\d+)', filename, re.IGNORECASE)
    cap_match = re.search(r'capacity-(\d+)MW', filename, re.IGNORECASE)

    site_id = int(site_match.group(1)) if site_match else np.nan
    nominal_capacity = float(cap_match.group(1)) if cap_match else np.nan

    if site_id == 3:
        print(f"⏭️ 건너뜀: {filename} (Site 3 제외)")
        continue

    df = pd.read_excel(file_path)
    df.rename(columns=col_mapping, inplace=True)

    # 시간 파싱
    time_col = 'Time(year-month-day h:m:s)'
    df[time_col] = pd.to_datetime(df[time_col])

    df['hour'] = df[time_col].dt.hour
    df['month'] = df[time_col].dt.month

    # 주기적 특성 인코딩
    df['sin(hour)'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos(hour)'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin(month)'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos(month)'] = np.cos(2 * np.pi * df['month'] / 12)

    # 용량 피처 추가
    df['Site ID'] = site_id
    df['Nominal capacity (MW)'] = nominal_capacity
    df['Maximum capacity (MW)'] = df['Power (MW)'].max()

    # 최종 컬럼 정렬
    target_cols = [
        time_col,
        'GHI', 'DNI', 'TSI', 'Temperature', 'Atmospheric pressure',
        'hour', 'sin(hour)', 'cos(hour)',
        'month', 'sin(month)', 'cos(month)',
        'Site ID', 'Nominal capacity (MW)', 'Maximum capacity (MW)',
        'Power (MW)'
    ]

    df_final = df[target_cols]

    output_filename = filename.replace('.xlsx', '.csv')
    output_path = os.path.join(output_dir, output_filename)
    df_final.to_csv(output_path, index=False)
    print(f"✅ CSV 저장 완료: {output_filename}")
