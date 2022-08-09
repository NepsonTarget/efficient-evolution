import json
import pandas as pd
import sapiens

seqs_abs = {
    'medi_vh': 'QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSYNAVWNWIRQSPSRGLEWLGRTYYRSGWYNDYAESVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARSGHITVFGVNVDAFDMWGQGTMVTVSS',
    'uca_vh': 'QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARGGHITIFGVNIDAFDIWGQGTMVTVSS',
    'mab114_vh': 'EVQLVESGGGLIQPGGSLRLSCAASGFALRMYDMHWVRQTIDKRLEWVSAVGPSGDTYYADSVKGRFAVSRENAKNSLSLQMNSLTAGDTAIYYCVRSDRGVAGLFDSWGQGILVTVSS',
    'mU_vh': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYDMHWVRQATGKGLEWVSAIGTAGDTYYPGSVKGRFTISRENAKNSLYLQMNSLRAGDTAVYYCVRSDRGVAGLFDSWGQGTLVTVSS',
    's309_vh': 'QVQLVQSGAEVKKPGASVKVSCKASGYPFTSYGISWVRQAPGQGLEWMGWISTYNGNTNYAQKFQGRVTMTTDTSTTTGYMELRRLRSDDTAVYYCARDYTRGAWFGESLIGGFDNWGQGTLVTVSS',
    'r7_vh': 'QVQLVESGGGVVQPGRSLRLSCAASGFTFSNYAMYWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRTEDTAVYYCASGSDYGDYLLVYWGQGTLVTVSS',
    'c143_vh': 'EVQLVESGGGLVQPGGSLRLSCAASGFSVSTKYMTWVRQAPGKGLEWVSVLYSGGSDYYADSVKGRFTISRDNSKNALYLQMNSLRVEDTGVYYCARDSSEVRDHPGHPGRSVGAFDIWGQGTMVTVSS',
    
    'medi_vl': 'DIQMTQSPSSLSASVGDRVTITCRTSQSLSSYTHWYQQKPGKAPKLLIYAASSRGSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSRTFGQGTKVEIK',
    'uca_vl': 'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSRTFGQGTKVEIK',
    'mab114_vl': 'DIQMTQSPSSLSASVGDRITITCRASQAFDNYVAWYQQRPGKVPKLLISAASALHAGVPSRFSGSGSGTHFTLTISSLQPEDVATYYCQNYNSAPLTFGGGTKVEIK',
    'mU_vl': 'DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKVPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQKYNSAPLTFGGGTKVEIK',
    's309_vl': 'EIVLTQSPGTLSLSPGERATLSCRASQTVSSTSLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQHDTSLTFGGGTKVEIK',
    'r7_vl': 'QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVSNRFSGSKSGNTASLTISGLQSEDEADYYCNSLTSISTWVFGGGTKLTVL',
    'c143_vl': 'QSALTQPASVSGSPGQSITISCTGTSNDVGSYTLVSWYQQYPGKAPKLLIFEGTKRSSGISNRFSGSKSGNTASLTISGLQGEDEADYYCCSYAGASTFVFGGGTKLTVL',
}

def eval_sapiens(seq, seq_name):
    scores = sapiens.predict_scores(seq, 'H' if seq_name.endswith('vh') else 'L')

    AAs = list(scores.columns)
    tok_to_idx = { aa: idx for idx, aa in enumerate(AAs) }

    data = []
    for pos in range(len(seq)):
        #seq_masked = seq[:pos] + '*' + seq[(pos + 1):]
        #scores = sapiens.predict_scores(seq_masked, 'H' if seq_name.endswith('vh') else 'L')
        wt = seq[pos]
        fracs = list(scores.iloc[[pos]].values[0])
        wt_frac = fracs[tok_to_idx[wt]]
        for frac, mt in zip(fracs, AAs):
            ratio = frac / wt_frac
            data.append([ pos + 1, wt, mt, 0, frac, ratio ])

    df = pd.DataFrame(data, columns=[
        'pos',
        'wt',
        'mt',
        'counts',
        'fraction',
        'likelihood_ratio',
    ])

    df.to_csv(f'target/sapiens/sapiens_likelihoods_{seq_name}.txt', sep='\t')

if __name__ == '__main__':
    for seq_name in seqs_abs:
        print(seq_name)
        seq = seqs_abs[seq_name]
        eval_sapiens(seq, seq_name)