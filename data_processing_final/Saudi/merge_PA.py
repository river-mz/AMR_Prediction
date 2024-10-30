import pandas as pd
import os

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

# 文件路径设置
PA_output_dir = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/PA'
atb_path = os.path.join(PA_output_dir, 'antibiotics_sep.csv')
PA_output_dir_tempt = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/PA_Sept_25'
atb_tempt_path = os.path.join(PA_output_dir_tempt, 'workflow_Sep_25.csv')

metaPath = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/final_PA_meta/meta'
sample_id_path = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/final_PA_meta/sample_id.csv'
sample_id_df = pd.read_csv(sample_id_path, header=0)
# sample_id_df = sample_id_df.rename(columns = {'Sample_ID': 'Sample ID'})

# 读取数据
atb_df = pd.read_csv(atb_path)
atb_tempt_df = pd.read_csv(atb_tempt_path)

print(atb_df.columns[:70])
print(atb_tempt_df.columns[:70])


atb_tempt_df = atb_tempt_df.rename(columns = {'Specimen': 'KAUST_ID'})
atb_tempt_df['KAUST_ID'] = atb_tempt_df['KAUST_ID'].str.replace(' ', '', regex=False).str.strip() 



print(atb_tempt_df[:2])
print(f"Original number of rows: {atb_tempt_df.shape[0]}")
atb_tempt_df = pd.merge(atb_tempt_df,sample_id_df,on = 'KAUST_ID', how = 'left' )
# Print the results
successful_merges = atb_tempt_df[atb_tempt_df['Sample_ID'].notnull()].shape[0]

print("merge atb with selected samples")

print(f"Number of rows after merging: {atb_tempt_df.shape[0]}")
print(f"Number of successfully merged rows: {successful_merges}")
# this

# atb_tempt_df['Specimen']  atb_df['Kaust samples ID']


# all_atb_df = pd.concat(atb_df, atb_tempt_df)


# 检查 atb_tempt_df['Specimen'] 与 atb_df['Kaust samples ID'] 之间的差异
specimen_set = set(atb_tempt_df['KAUST_ID'].dropna())
kaust_id_set = set(atb_df['Kaust samples ID'].dropna())



all_date_path = os.path.join(PA_output_dir_tempt, 'All PA samples.csv')
all_date_df = pd.read_csv(all_date_path)

sample_merged_date_df = all_date_df

# date

print(sample_merged_date_df.columns)
# sample_merged_date_set = set(sample_merged_date_df['KAUST_ID'])



print(sample_merged_date_df.shape)
# atb_date_df = pd.merge(sample_merged_date_df, atb_df,  left_on = 'KAUST_ID', right_on='Kaust samples ID', how = 'left')
atb_date_sample_df = pd.merge(sample_merged_date_df, atb_df, left_on='Sample ID', right_on = 'Sample_ID', how = 'left')
successful_merges = atb_date_sample_df[atb_date_sample_df['Sample_ID'].notnull()].shape[0]

atb_date_sample_df = pd.merge(sample_merged_date_df, atb_tempt_df, left_on='Sample ID', right_on = 'Sample_ID', how = 'left')
successful_merges2 = atb_date_sample_df[atb_date_sample_df['Sample_ID'].notnull()].shape[0]

# Print the results
print("merge atb with selected samples")
print(f"Original number of rows: {sample_merged_date_df.shape[0]}")
print(f"Number of rows after merging: {atb_date_sample_df.shape[0]}")
print(f"Number of successfully merged 1 rows: {successful_merges}")
print(f"Number of successfully merged 2 rows: {successful_merges2}")
# atb_date_df = atb_date_df.drop_duplicates(subset=['Sample_ID'], keep='first')



# print(atb_df['Sample_ID'][:5])
# print(atb_df['Kaust samples ID'][:5])
# print(atb_tempt_df['Specimen'][:5])




merged_df = atb_date_sample_df
# merged_df = merged_df.rename(columns = {})
print(merged_df.columns)
# print(merged_df['Date of Collection'][:5])
# print(atb_tempt_df['Received Date'][:5])

# ----
# merge with op/ip
merged_df['AST_id'] = merged_df.index
merged_df['Date of Collection'] = pd.to_datetime(merged_df['Date of Collection'], dayfirst=True, errors='coerce')
print(merged_df['Date of Collection'])
# dir_path = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/PA_Sept_25'
ip_df = pd.read_csv(os.path.join(metaPath, 'IP visits.csv'),header=0)
# print(ip_df.columns)
ip_df['DS_DT'] = pd.to_datetime(ip_df['DS_DT'])
ip_df['ADS_DT'] = pd.to_datetime(ip_df['ADS_DT'])

ip_df.columns = [v+'_ip' for v in ip_df.columns]
ip_df['PT_NO_ip'] = [str(x) for x in ip_df['PT_NO_ip']]
# print(merged_df[])
print(merged_df.columns[:100])
merged_ip_df = pd.merge(
    merged_df,
    ip_df,
    left_on = 'PT_NO.x',
    right_on = 'PT_NO_ip',
    how='inner'                  # Use inner join
)

# print(merged_ip_df[merged_ip_df['PACT_ID_ip'].notnull()].shape[0])

merged_ip_df = merged_ip_df[
    (merged_ip_df['DS_DT_ip'] > merged_ip_df['Date of Collection']) &
    (merged_ip_df['ADS_DT_ip'] < merged_ip_df['Date of Collection'])
]
# merge merged_df with ip_df according to 'PACT_ID.x_x'( merged_df) , 'PACT_ID.x_x'( merged_df) 

print('ip merged filtered: ')
print(merged_ip_df.shape)

merged_ip_df = pd.concat([merged_ip_df, merged_df])
merged_ip_df = merged_ip_df.drop_duplicates(keep = 'first', subset = 'AST_id')
print("go back")
print(merged_ip_df.shape)


#merge with op
print("merge op")

# dir_path = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/PA_Sept_25'
op_df = pd.read_csv(os.path.join(metaPath, 'OP visits.csv'),header=0)
# op_df['MED_DT'] = op_df['MED_DT'].replace('4017-04-01', '2017-04-01')
def correct_out_of_bounds_dates(date):
    try:
        return pd.to_datetime(date)
    except pd.errors.OutOfBoundsDatetime:
        # 如果日期超过合理范围，将年分错误的 4017 修正为 2017
        if '4017' in date:
            return pd.to_datetime(date.replace('4017', '2017'))
        else:
            return pd.NaT  # 将无法修复的日期设为 NaT

# 应用函数修正日期列
# op_df['MED_DT'] = op_df['MED_DT'].apply(correct_out_of_bounds_dates)
op_df['MED_DT'] = op_df['MED_DT'].str.replace('4017', '2017')
op_df['MED_DT'] = op_df['MED_DT'].str.replace(' 00:00:00', '')
print(op_df.columns)
op_df['MED_DT'] = pd.to_datetime(op_df['MED_DT'])
print(op_df.shape)
op_df = op_df.drop_duplicates(keep='first')
print(op_df.shape)
op_df.columns = [v+'_op' for v in op_df.columns]
op_df['PT_NO_op'] = [str(x) for x in op_df['PT_NO_op']]

op_df = op_df[op_df['PT_NO_op'].isin(merged_ip_df['PT_NO.x'])]
print(op_df.shape)
# print(merged_df[])
merged_ip_op_df = pd.merge(
    merged_ip_df,
    op_df,
    left_on = 'PT_NO.x',
    right_on = 'PT_NO_op',
    how='inner'                  # Use inner join
)
print('merge op successfully')
print(merged_ip_op_df[merged_ip_op_df['PACT_ID_op'].notnull()].shape[0])
print('match ip')
print(len(set(merged_ip_op_df['PT_NO.x']) - set(op_df['PT_NO_op'])))
merged_ip_op_df = merged_ip_op_df[
     (merged_ip_op_df['MED_DT_op'] - merged_ip_op_df['Date of Collection']).abs().dt.days < 7]
# merge merged_df with ip_df according to 'PACT_ID.x_x'( merged_df) , 'PACT_ID.x_x'( merged_df) 

print('op merged filtered: ')
print(merged_ip_op_df.shape)

merged_ip_op_df = pd.concat([merged_ip_op_df, merged_ip_df])
merged_ip_op_df = merged_ip_op_df.drop_duplicates(keep = 'first', subset = 'AST_id')
print('op go back')
print(merged_ip_op_df.shape)



successful_merges_ip = merged_ip_op_df[merged_ip_op_df['ADS_DT_ip'].notnull()].shape[0]
successful_merges_op = merged_ip_op_df[merged_ip_op_df['MED_DT_op'].notnull()].shape[0]

successful_merges_ip_op = merged_ip_op_df[merged_ip_op_df['MED_DT_op'].notnull() & merged_ip_op_df['ADS_DT_ip'].notnull()].shape[0]

print(successful_merges_ip)
print(successful_merges_op)
print(successful_merges_ip_op)
# op_df = pd.read_csv(os.path.join(dir_path, 'OP Visits.csv'),header=0)
# print(op_df.columns)

# ip_op_df = pd.concat([ip_df, op_df])
# # print(ip_df.shape)
# # print(op_df.shape)
# # print(ip_op_df.shape)
# ip_op_df['PACT_ID'] = ip_op_df['PACT_ID'].apply(lambda x: str(x))

# print(len(set(filtered_merged_df['PACT_ID_dx'])-set(ip_op_df['PACT_ID'])))
# print(len(set(filtered_merged_df['PACT_ID_dx'])))

# ip_op_df.columns = [v+'_ip_op' for v in ip_op_df.columns]


# filtered_merged_ipop_df = pd.merge(
#     filtered_merged_df,
#     ip_op_df,
#     left_on = 'PACT_ID_dx',
#     right_on = 'PACT_ID_ip_op',
#     how='left'                  # Use inner join
# )

# print(filtered_merged_ipop_df.shape)
# print(filtered_merged_df.shape)

# # Calculate the number of successfully merged rows
# successful_merges = filtered_merged_ipop_df[filtered_merged_ipop_df['PACT_ID_ip_op'].notnull()].shape[0]

# # Print the results
# print(f"Original number of rows: {filtered_merged_df.shape}")
# print(f"Number of rows after merging: {filtered_merged_ipop_df.shape}")
# print(f"Number of successfully merged rows: {successful_merges}")




# # merge with dx

print('merge dx')

# dir_path = '/ibex/project/c2205/AMR_dataset_peijun/Saudi/PA_Sept_25'
dx_df = pd.read_csv(os.path.join(metaPath, 'All Dx..csv'),header=0)
print(dx_df.columns)

dx_df.columns = [v+'_dx' for v in dx_df.columns]

dx_df['PT_NO_dx'] = dx_df['PT_NO_dx'].apply(lambda x: str(x))

# add a index column to merged_df
merged_df = merged_ip_op_df 


merged_ip_op_with_dx = pd.merge(
    merged_ip_op_df,
    dx_df,
    left_on = 'PT_NO.x',
    right_on = 'PT_NO_dx',
    how='inner'                  # Use inner join
)

# Check the number of rows before and after merging
original_row_count = merged_ip_op_df.shape[0]
merged_row_count = merged_ip_op_with_dx.shape[0]

# Calculate the number of successfully merged rows
successful_merges = merged_ip_op_with_dx[merged_ip_op_with_dx['PT_NO_dx'].notnull()].shape[0]

# Print the results
print(f"Original number of rows: {original_row_count}")
print(f"Number of rows after merging: {merged_row_count}")
print(f"Number of successfully merged rows: {successful_merges}")


# 确保 'Date of Collection' 和 'DGNS_REG_DT_dx' 是 datetime 类型
merged_ip_op_with_dx['Date of Collection'] = pd.to_datetime(merged_ip_op_with_dx['Date of Collection'], dayfirst=True, errors='coerce')
merged_ip_op_with_dx['DGNS_REG_DT_dx'] = pd.to_datetime(merged_ip_op_with_dx['DGNS_REG_DT_dx'])

# 计算两列的日期差
merged_ip_op_with_dx['date_diff'] = (merged_ip_op_with_dx['Date of Collection'] - merged_ip_op_with_dx['DGNS_REG_DT_dx']).abs()

# 筛选日期差小于14天的行
filtered_df = merged_ip_op_with_dx[merged_ip_op_with_dx['date_diff'] <= pd.Timedelta(days=14)]

# 去掉 'date_diff' 列，如果不需要的话
filtered_df = filtered_df.drop(columns=['date_diff'])
print(len(set(filtered_df['PT_NO_dx'])))
print(len(set(merged_ip_op_df['PT_NO.x'])))
# 输出结果
print(filtered_df.shape)


filtered_merged_df = pd.concat([filtered_df, merged_ip_op_df])
filtered_merged_df = filtered_merged_df.drop_duplicates(keep = 'first', subset = 'AST_id')
print(filtered_merged_df.shape)


# merge with demo

import pandas as pd
import os

# Step 1: Load the CSV file

demo_df = pd.read_csv(os.path.join(metaPath, 'Demo..csv'), header=0)
demo_df = demo_df.drop_duplicates(keep = 'first')
# remove duplicate row with same pt_id
demo_df = demo_df.drop_duplicates(subset = 'PT_NO', keep = False)

# Step 2: Convert PT_NO to string
demo_df['PT_NO'] = demo_df['PT_NO'].apply(lambda x: str(x))

# Step 3: Rename columns with suffix _demo
demo_df.columns = [f"{v}_demo" for v in demo_df.columns]

filtered_merged_ipop_demo_df = pd.merge(
    filtered_merged_df,
    demo_df,
    left_on = 'PT_NO.x',
    right_on = 'PT_NO_demo',
    how='left'                  # Use inner join
)

print(filtered_merged_df.shape)
print(demo_df.shape)

print("Successfully merge with demo:")
print(filtered_merged_ipop_demo_df[filtered_merged_ipop_demo_df['PT_NO_demo'].notnull()].shape[0])

filtered_merged_ipop_demo_df.to_csv(os.path.join(metaPath, 'result2.csv'), index=0)
print(filtered_merged_ipop_demo_df.columns[-50:])

etpr_columns = [col for col in filtered_merged_ipop_demo_df.columns if 'ETPR_PT_NO' in col]

# Create a new DataFrame with only the selected columns
etpr_df = filtered_merged_ipop_demo_df[etpr_columns]

# Save the new DataFrame to a CSV file (optional)
etpr_df.to_csv('etpr_columns_data.csv', index=False)
equal_rows = etpr_df[(etpr_df.nunique(axis=1) == 1) | (etpr_df.isna().all(axis=1))]
print(f"具有相同值的行数: {equal_rows.shape}")
print(etpr_df.shape)
# print(equal_rows)
different_rows = etpr_df[~etpr_df.index.isin(equal_rows.index)]

print("不相同的行:")
print(different_rows)
