import os
import matplotlib.pyplot as plt

MARKERS = ['s','o','d','p','^',',','.','v','<','>','1','2','3','4','8','*','P','h','H','+','x','X','D','|','_']
LINE_STYLES = ['-', '--', '-.', ':']
COLORS = ['#65a9d7', '#e9963e', '#f23b27', '#304f9e', '#449945']

def plot(save_path: str="./figures", file_name: str="performances", linewidth: float=1.0):
    # make the length of x,y 
    plt.figure(figsize=(6,6))

    # set the font of the scale 
    plt.tick_params(labelsize=13)
    
    x_index = ["T5-base", "T5-large", "T5-3B"]
    # for generator
    mix_results = [33.1094, 38.75, 46.0469]
    no_mix_results = [33.375, 40.0312, 42.25]
    plt.plot(x_index, mix_results, color=COLORS[0], linestyle=LINE_STYLES[1], marker=MARKERS[1], label="Generator (w/ GEN)", linewidth=linewidth)
    plt.plot(x_index, no_mix_results, color=COLORS[2], linestyle=LINE_STYLES[1], marker=MARKERS[1], label="Generator (w/o GEN)", linewidth=linewidth)
    
    # for classifier
    mix_results = [40.3594, 43.0156, 50.1562]
    no_mix_results = [40.4844, 44.2812, 48.7656]
    plt.plot(x_index, mix_results, color=COLORS[0], linestyle=LINE_STYLES[0], marker=MARKERS[4], label="Classifier (w/ GEN)", linewidth=linewidth)
    plt.plot(x_index, no_mix_results, color=COLORS[2], linestyle=LINE_STYLES[0], marker=MARKERS[4], label="Classifier (w/o GEN)", linewidth=linewidth)
    

    # set the font of xlabel and ylabel
    font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
    }
    # plt.xlabel("Model with different size",font)
    plt.ylabel("EM score",font)


    legend = plt.legend(loc="upper left", frameon=True,shadow=True, fontsize='small') # x-large)

    # remove the margin around the pdf  
    plt.tight_layout()  
    plt.grid(axis = 'both', linestyle='-.', linewidth=0.5, color='gray', alpha=0.5)

    # save the pdf
    save_file = "{}.png".format(file_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, save_file)) 
    plt.show() 
    print("save the plot to {}\n".format(os.path.join(save_path, save_file)))


if __name__ == "__main__":
    plot(save_path="./figures", file_name="performances", linewidth=1.0)