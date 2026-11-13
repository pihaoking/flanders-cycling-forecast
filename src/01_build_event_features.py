"""
Build event features (public + school holidays) for each date in
2019-08-01 to 2025-12-31. Saves to data/event_features.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

# date range matching cycling data
dates = pd.date_range('2019-08-01', '2025-12-31', freq='D')
df = pd.DataFrame({'date': dates})

# public holidays
hol = pd.read_csv(DATA / 'belgian_holidays.csv')
hol['date'] = pd.to_datetime(hol['date'])
df['is_public_holiday'] = df['date'].isin(hol['date']).astype(int)

# Holiday type mapping (Dutch name to category)
type_map = {
    'Nieuwjaar': 'new_year',
    'Pasen': 'easter', 'Paasmaandag': 'easter',
    'Dag van de Arbeid': 'national',
    'O. L. H. Hemelvaart': 'national',
    'Pinksteren': 'national', 'Pinkstermaandag': 'national',
    'Nationale feestdag': 'national',
    'O. L. V. Hemelvaart': 'national',
    'Allerheiligen': 'national',
    'Wapenstilstand': 'national',
    'Kerstmis': 'christmas',
}
hol['holiday_type'] = hol['holiday_name'].map(type_map).fillna('other')
hol_type = hol[['date', 'holiday_type']].drop_duplicates('date')
df = df.merge(hol_type, on='date', how='left')

# school holidays
sch = pd.read_csv(DATA / 'school_holidays_flanders.csv')
sch['start_date'] = pd.to_datetime(sch['start_date'])
sch['end_date'] = pd.to_datetime(sch['end_date'])

df['is_school_holiday'] = 0
for _, row in sch.iterrows():
    mask = (df['date'] >= row['start_date']) & (df['date'] <= row['end_date'])
    df.loc[mask, 'is_school_holiday'] = 1
    # Also fill holiday_type for school holiday dates (if not already a public holiday)
    df.loc[mask & df['holiday_type'].isna(), 'holiday_type'] = row['holiday_type']

# Fill remaining NaN holiday_type as 'none'
df['holiday_type'] = df['holiday_type'].fillna('none')

# days to nearest public holiday: |date - hol| matrix, min per row
hol_ts = hol['date'].values.astype('datetime64[D]').astype(np.int64)
date_ts = df['date'].values.astype('datetime64[D]').astype(np.int64)
df['days_to_nearest_holiday'] = np.abs(date_ts[:, None] - hol_ts[None, :]).min(axis=1)

# summary
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total days: {len(df)}")
print(f"Public holiday days: {df['is_public_holiday'].sum()}")
print(f"School holiday days: {df['is_school_holiday'].sum()}")
print(f"\nHoliday type distribution:")
print(df[df['holiday_type'] != 'none']['holiday_type'].value_counts())
print(f"\nDays to nearest holiday stats:")
print(df['days_to_nearest_holiday'].describe())

# save
df.to_csv(DATA / 'event_features.csv', index=False)
print(f"\nSaved to {DATA / 'event_features.csv'}")
