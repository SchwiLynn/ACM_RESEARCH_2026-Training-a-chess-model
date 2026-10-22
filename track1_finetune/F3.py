import pandas as pd
import time
from maia2 import model, inference

def maia2_predict(data, device="cpu", results="results.csv"):
    print(f"Starting Baseline Evaluation on {device.upper()}")
    maia2_model=model.from_pretrained(type="rapid", device=device)
    start_time=time.time()
    total_positions=len(data)
    print(f"Processing {total_positions} positions")
    df_results, _=inference.inference_batch(data, maia2_model, verbose=1, batch_size=1024, num_workers=0)
    output_data=[]
    for move_probs in df_results['move_probs']:
        sorted_moves=sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        top_3=sorted_moves[:3]
        top_3+=[(None, 0.0)]*(3-len(top_3))
        
        output_data.append(
        {
            'prediction1':top_3[0][0], 'probability1':top_3[0][1],
            'prediction2':top_3[1][0], 'probability2':top_3[1][1],
            'prediction3':top_3[2][0], 'probability3':top_3[2][1]
        })
    df_results=pd.concat([df_results, pd.DataFrame(output_data)], axis=1)
    time_taken=time.time()-start_time
    positions_per_sec=(total_positions/time_taken) if time_taken>0 else 0
    if 'move' in df_results.columns:
        top1_correct=(df_results['prediction1']==df_results['move']).sum()
        top3_correct=df_results[['prediction1', 'prediction2', 'prediction3']].eq(df_results['move'], axis=0).any(axis=1).sum()
        top1_acc=(top1_correct/total_positions)*100
        top3_acc=(top3_correct/total_positions)*100
    else:
        top1_acc, top3_acc=0, 0
    df_results.to_csv(results, index=False)
    print(f"\n{device.upper()} | {total_positions} pos | {positions_per_sec:.1f} pos/sec | Top-1: {top1_acc:.1f}% | Top-3: {top3_acc:.1f}%")
    return df_results

if __name__=="__main__":
    print("Loading exampleGame.csv")
    df_real_data=pd.read_csv("track1_finetune/exampleGame.csv")
    maia2_predict(df_real_data, device="cpu", results="results.csv")