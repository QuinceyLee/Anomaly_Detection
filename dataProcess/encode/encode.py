import pandas as pd
from dataProcess.conn2csv import filed
from utils.find_file import find_all_file_csv

input_names = ['ts',
               "id.orig_p",
               "id.resp_p",
               "proto",
               "service",
               "duration",
               "orig_bytes",
               "resp_bytes",
               "conn_state",
               "missed_bytes",
               "history",
               "orig_pkts",
               "orig_ip_bytes",
               "resp_pkts",
               "resp_ip_bytes",
               "detailed-label"
               ]
# pd_dic = {
#     "id.orig_p":int,
#     "id.resp_p":int,
#     "proto":int,
#     "service":int,
#     "duration":float,
#     "orig_bytes":np.long,
#     "resp_bytes":np.long,
#     "conn_state":int,
#     "missed_bytes":np.long,
#     "history":int,
#     "orig_pkts":np.long,
#     "orig_ip_bytes":np.long,
#     "resp_pkts",
#     "resp_ip_bytes",
#     "detailed-label"
# }
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
data_csv = '/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/csv_2'


def convert_label(df):
    his_list = [
        '-1',
        'S',
        'D',
        'Dd',
        'Sr',
        'ShAdDafF',
        'ShADadfF',
        'ShAdDaFf',
        'ShADafF',
        'ShADar',
        'ShAdDaFr',
        'ShAdDfFr',
        'R',
        'ShADr',
        'ShADdfFa',
        'ShAdDaftF',
        'ShAdDr',
        'ShADafr',
        'ShAdDafrR',
        'ShAdDarfR',
        'ShADfFa',
        'ShAdDfFa',
        'ShADfFr',
        'ShAdDafFr',
        '^r',
        'ShADfrFr',
        'ShADfaF',
        'ShAdDatFr',
        'ShAfFa',
        'ShAr',
        'ShADadfR',
        'ShA',
        'ShAdDtafF',
        'F',
        'Ar',
        'ShAF',
        'ShAdDatfF',
        'ShADafdtF',
        'ShAdtDaFr',
        'D^',
        'HaDdAfF',
        'ShAdDaFfR',
        'HaDdAFf',
        'ShAdDatFrR',
        'ShAdDaTFf',
        'ShAdDaFfr',
        'ShAdtDaFrR',
        'ShAdtDafF',
        'ShAdDaFrR',
        'ShAdfFa',
        'ShAdDafFR',
        'ShAdDaF',
        'SaR',
        'ShADadR',
        'ShAFr',
        'ShADadRf',
        'ShADF',
        'ShAdDatrfR',
        'ShAFa',
        'ShAdDaTfF',
        'Fa',
        'ShADFr',
        'ShAdDaftFR',
        'HaDdR',
        'ShAadDfF',
        'ShAfdtDFr',
        'ShAdDaFTf',
        'FaR',
        'ShAdDaFRf',
        'ShADadFf',
        'ShAdDtafFr',
        'ShAdDar',
        'ShAdDarr',
        'ShADfFrr',
        'ShAdaFr',
        'FaAr',
        'HaADdFf',
        'ShAdDFar',
        'ShAdDafFrr',
        'ShAdfDFr',
        'ShAdDafrFr',
        'ShAFfR',
        'ShADfdtR',
        'ShAdDaFRRf',
        'HaR',
        'HaDdTAFf',
        '^aR',
        'ShAdDafr',
        'HafFr',
        'ShADfF',
        '^hADadfR',
        'ShADfR',
        'ShADfdtFaR',
        '^hA',
        'Ffa',
        'ShADfrF',
        'ShAFdfRt',
        '^hADFr',
        'ShAFdRfR',
        'ShAdDTafF',
        'ShAdDFfR',
        '^hADafF',
        'ShADarfF',
        'ShAdDaTF',
        'ShADFadfRR',
        'ShAdDatFf',
        'ShADrfR',
        'ShADdafR',
        'ShAFf',
        'ShADaF',
        'ShAdF',
        'ShAdDafFrR',
        'SahAdDrfR',
        'Fr',
        'ShAdDaTFfR',
        'ShAdFaf',
        'ShrA',
        'ShAdr',
        'ShAdDtaFr',
        'HaFfA',
        'ShAdDaFRfR',
        'ShADdfR',
        'FfA',
        'ShAFafR',
        'HaDdAFTf',
        '^hADr',
        'ShAdDaFRR',
        'ShAdDaFfRR',
        'ShAdDaFRRfR',
        'ShAa',
        'ShAdDaR',
        'ShAadDFf',
        'SahAdDFf',
        'ShAdDaFR',
        'ShAdDaFRfRR',
        '^aA',
        'ShADFar',
        'ShADad',
        'ShAdDFaf',
        'ShADaR',
        'ShADa',
        'ShADFa',
        'ShAdDaFT',
        'SahAdDtFf',
        'SahAdDFRf',
        'SahAdDF',
        'DFafA',
        'ShAdDaTFRf',
        'ShAadDFRf',
        'ShAdDtaFf',
        'ShADFadfR',
        'ShAadDFR',
        'ShAdDFf',
        '^d',
        'ShAafF',
        'ShADFfa',
        'ShAdDatR',
        'ShADadFfR',
        'ShArR',
        'ShADFaR',
        'ShAdDaFRRRf',
        'ShR',
        'ShADFfR',
        'ShAadDr',
        'DdA',
        'ShADadfrr',
        'ShAdDaFRr',
        'ShADadtfF',
        'ShAFaf',
        'ShADacfgF',
        'ShADacgdFf',
        'ShADCaGdfF',
        'ShADCaGdFf',
        'ShADadfr',
        'AaDd',
        'ShADadfFr',
        'ShADadf',
        'DRr',
        'ShADadr',
        'ShADadfRr',
        'ShAaDdFf',
        'ShAFfa',
        'DadA',
        'ShADadRRR',
        'ShADadttWWtf',
        'ShADadftF',
        'ShADadtftr',
        'ShADadtf',
        'ShADadFfRR',
        'DdAtfFa',
        'ShADadtFRfR',
        'ShADadft',
        'ShADadtfRr',
        '^f',
        'ShDadAttt',
        'Hr',
        'A',
        'ShADFrfR',
        'ShADFadRfR',
        'ShADadfFR',
        '^ha',
        'DdAa',
        'DdAf',
        'H',
        'ShADaFr',
        'ShADadtFf',
        'ShADdFaf',
        '^hR',
        'ShAdDtaR',
        'SAD',
        'ShADadtRf',
        'ShADadftFR',
        'ShAdDaTR',
        'ShADdtaFf',
        'ShADdFf',
        'ShAdDafR',
        'DAd',
        'HWr',
        '^hwR',
        'ShwR',
        'HrR',
        '^hRr',
        '^hRRRRRaR',
        'ShADFafdtRR',
        'D^d',
        'ShADFadRf',
        'DdAFaf',
        '^hRaRr',
        'ShADaCGdtfF',
        'ShADaCGr',
        'ShADCaGcgd',
        '^dDA',
        'ShADaCGcgdF',
        'ShADaCGdt',
        'ShADFafdtR',
        'ShADFaf',
        'ShwA',
        'ShADFafR',
        'ShADFaTdfR',
        'ShADFaTdRf',
        'ShADFfr',
        'ShwAr',
        'ShAFar',
        'ShADFaTfdtR',
        'ShADFadRtf',
        'ShADFdfR',
        'ShADFadR',
        'ShADFaTf',
        'ShADFfaRR',
        'ShADFafRdt',
        'ShADFdafRR',
        'ShADFfRaR',
        'ShADFadtRf',
        'ShADFadftR',
        'ShAD',
        'ShAFfar',
        'ShAFafr',
        'HadfDrArR',
        'ShADFdfaRR',
        'ShADFfdtaRR',
        'ShwAaFdfR',
        'ShADFadRftr',
        'ShADFarR',
        'ShADFaTdftR',
        'ShADFrRfaR',
        'Aa',
        'ShADFdRf',
        'ShADFdRafR',
        'ShADFaT',
        'ShADFaTr',
        'DaFfA',
        'ShADFadfRt',
        'C',
        'ShAdDaf',
        'ShAdDaft',
        'ShAdfDr',
        'CCCC',
        'ShADadtcfF',
        'ShADadttcfF',
        'ShAdDatfr',
        'CCC',
        'ShDadAf',
        'ShAfdtDr',
        'ShADacdtfF',
        'ShADadtctfF',
        'ShAdDatf',
        'ShADadttfF',
        'ShAdD',
        'ShADadtctfFR',
        'ShAdDfr',
        'DdAtaFf',
        'ShAdDa',
        'I',
        'DTT',
        'SI',
        'ShADdattFfR',
        'ShAfF',
        'ShADdfF',
        'ShAdfr',
        'ShAdaDR',
        'ShAaw',
        'Dr',
        'HaDdAr',
        'ShAdDaTRf',
        'ShAdDaRRR',
        'ShADdf',
        'ShAfdtF',
        'ShAdfF',
        'ShADdtatFfR',
        'ShAdtfFa',
        'DFr',
        'DrF',
        'DdAaFf',
        'ShAdDaTRr',
        'ShAdfDF',
        'ShAdDaTFR',
        'ShAdDaRr',
        'ShAdDaTfRr',
        'DT',
        '^dtt',
        'ShAdDaTRft',
        'ShADFdRtf',
        'ShADFdRt',
        'ShADdafF',
        'ShwAadDfF',
        'ShADdaFf',
        'ShADadCFf',
        'ShADdfr',
        'ShADdfFr',
        'ShADfr',
        'ShADdaftF',
        '^fA',
        'ShADdaCFf',
        'ShADdacFf',
        'ShAaww',
        'ShADadFRf',
        'ShADadF',
        'ShADadcFf',
        'ShADadCcFf',
        'ShADacfF',
        'ShADadFRR',
        'ShADadTFTf',
        'ShADdatcFf',
        'ShADdaFr',
        'ShADdFfa',
        'ShADfrr',
        'ShADadttFf',
        'ShADdaTFf',
        'ShADadtTfFr',
        'ShADacdtfr',
        'ShAdaw',
        '^hwAadDfF',
        'ShADadtfFr',
        'ShwAadDftF',
        'ShADacdtttfF',
        'ShADacdftF',
        'ShADadTFf',
        'ShwAadDfr',
        'ShADdFfaRR',
        'ShADadFRfR',
        'ShADdCacFf',
        'ShADacdFf',
        'ShADadCtFf',
        'ShADadttFRfR',
        'ShADadCfF',
        'ShADadtCFf',
        'ShADdTafF',
        'ShADdaf',
        'ShADafFr',
        'ShADadtR',
        'DdAttfrF',
        'ShADaTdR',
        'ShADdR',
        'ShADaTdtR',
        'ShADaTfF',
        'ShADda',
        'ShAfFr',
        'ShArr',
        'ShAar',
        'ShAdDaTfR',
        'ShAdDaTTRf',
        'ShAdFaRf',
        'ShAdDatRRR',
        'ShAdDR',
        'ShAdDaRR',
        'ShAfdtFa',
        'ShAfr',
        'ShAdDaTfr',
        'ShADdfrFr',
        'ShADfdtrFr',
        'ShADdrfFr',
        'ShAdfR',
        'ShADard',
        'ShAfdtR',
        'ShADfdtFr',
        'ShADdfFrr',
        'ShADadtftF',
        'ShADdar',
        'ShADadcgttfF',
        'ShADadcgtftF',
        '^c',
        'ShAdDaT']
    num_list = range(0, 402)
    history_dic = dict(zip(his_list, num_list))
    label_dic = {
        -1: 0, -1.0: 0, '-1': 0, '-1.0': 0,
        'C&C': 1, 'C&C-Torii': 1, 'C&C-Mirai': 1,
        'PartOfAHorizontalPortScan': 2, 'C&C-PartOfAHorizontalPortScan': 2,
        'PartOfAHorizontalPortScan-Attack': 2,
        'Attack': 3,
        'Okiru': 4, 'Okiru-Attack': 4,
        'DDoS': 5,
        'FileDownload': 6, 'C&C-FileDownload': 6,
        'C&C-HeartBeat': 7, 'C&C-HeartBeat-FileDownload': 7, 'C&C-HeartBeat-Attack': 7
    }
    proto_dic = {
        'tcp': 0,
        'udp': 1,
        'icmp': 2
    }
    service_dic = {
        '-1': 0, '-1.0': 0, ' ': 0, -1: 0, -1.0: 0,
        'http': 1,
        'dns': 2,
        'dhcp': 3,
        'ssl': 4,
        'ssh': 5,
        'irc': 6
    }
    conn_list = [
        'SF',
        'S0',
        'RSTR',
        'OTH',
        'SH',
        'S3',
        'S1',
        'SHR',
        'RSTO',
        'RSTRH',
        'REJ',
        'RSTOS0',
        'S2'
    ]
    normal_dic = {
        -1: 0, -1.0: 0, ' ': 0, '-1': 0, '-1.0': 0
    }
    conn_dic = dict(zip(conn_list, num_list))
    new_df = df.replace({'history': history_dic})
    new_df = new_df.replace({'detailed-label': label_dic})
    new_df = new_df.replace({'proto': proto_dic})
    new_df = new_df.replace({'service': service_dic})
    new_df = new_df.replace({'conn_state': conn_dic})
    new_df = new_df.replace({'duration': normal_dic})
    new_df = new_df.replace({'orig_bytes': normal_dic})
    new_df = new_df.replace({'resp_bytes': normal_dic})
    new_df = new_df.replace({'resp_bytes': normal_dic})
    new_df = new_df.replace({'missed_bytes': normal_dic})
    new_df = new_df.replace({'orig_pkts': normal_dic})
    new_df = new_df.replace({'orig_ip_bytes': normal_dic})
    new_df = new_df.replace({'resp_pkts': normal_dic})
    new_df = new_df.replace({'resp_ip_bytes': normal_dic})
    new_df.drop('label', axis=1, inplace=True)
    new_df.drop('tunnel_parents', axis=1, inplace=True)
    return new_df


def save_encode(name, file_fold, out_fold):
    print(name + " start")
    temp = pd.read_csv(file_fold + '/' + name, header=None, names=filed)
    new = convert_label(temp)
    new.to_csv(out_fold + '/' + name, header=None, index=None)
    print(name + " finished")


def save_encode_dir(file_fold, out_fold):
    for i in find_all_file_csv(file_fold):
        save_encode(i, file_fold, out_fold)


def merge(fold):
    with open('../../test/test.csv', 'ab') as f:
        for item in find_all_file_csv(fold):
            f.write(open(fold + '/' + item, 'rb').read())


# collect_sample('/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/csv/sta')
# save_encode_dir(data_csv, '/Volumes/T7 Touch/毕设相关/安全检测/数据集/opt/encode_5')
merge('../../merge')
