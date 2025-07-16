#!/usr/bin/env python3
"""
Generate compressed Gamburyan (with UFC) transcript visualization
"""

import sys
sys.path.append('.')
from generate_compressed_viz import create_transcript_visualization
from pathlib import Path

# Gamburyan data with UFC (98 tokens)
gamburyan_with_ufc_data = {
    'original_text': 'Gamburyan is set to make his 135-pound men\'s bantamweight debut against Cody "The Renegade" Gibson at the MGM Grand Arena on a pay-per-view card headlined by the UFC light heavyweight title fight between champion Jon Jones and Daniel Cormierr',
    'total_tokens': 98,
    'tokens': [
        {'position': 0, 'token': 'Looking', 'explanation': 'Referred oun vacated Ging[Looking]', 'kl_divergence': 4.554711, 'mse': 11510.502930, 'relative_rmse': 0.987623},
        {'position': 1, 'token': 'to', 'explanation': 'Available Looking SearchLooking[ to]', 'kl_divergence': 0.470059, 'mse': 2.230685, 'relative_rmse': 0.581989},
        {'position': 2, 'token': 'leave', 'explanation': 'Browse desires To Expect[ leave]', 'kl_divergence': 0.315791, 'mse': 1.931774, 'relative_rmse': 0.435581},
        {'position': 3, 'token': 'a', 'explanation': 'Leave leave Leave leave[ a]', 'kl_divergence': 0.208877, 'mse': 1.457611, 'relative_rmse': 0.392180},
        {'position': 4, 'token': 'two', 'explanation': 'Reserved saved created placed[ two]', 'kl_divergence': 0.106378, 'mse': 1.450951, 'relative_rmse': 0.382455},
        {'position': 5, 'token': '-', 'explanation': 'Reserved two opted nested[-]', 'kl_divergence': 0.342930, 'mse': 1.632013, 'relative_rmse': 0.352771},
        {'position': 6, 'token': 'fight', 'explanation': 'Defensive two relocated six[fight]', 'kl_divergence': 1.585555, 'mse': 1.796041, 'relative_rmse': 0.379466},
        {'position': 7, 'token': 'win', 'explanation': 'Played competitive sealed wrestle[ win]', 'kl_divergence': 2.332783, 'mse': 2.065120, 'relative_rmse': 0.415632},
        {'position': 8, 'token': 'less', 'explanation': 'Playoffs win victory win[less]', 'kl_divergence': 2.568291, 'mse': 1.530998, 'relative_rmse': 0.414899},
        {'position': 9, 'token': 'streak', 'explanation': 'Playoffs undefeated finish undefeated[ streak]', 'kl_divergence': 2.052547, 'mse': 2.031145, 'relative_rmse': 0.417870},
        {'position': 10, 'token': 'behind', 'explanation': 'streak goodbye trophy leave[ behind]', 'kl_divergence': 3.397440, 'mse': 1.885765, 'relative_rmse': 0.450470},
        {'position': 11, 'token': 'him', 'explanation': 'matchups unsustainable backfield atop[ him]', 'kl_divergence': 2.231226, 'mse': 1.982691, 'relative_rmse': 0.419812},
        {'position': 12, 'token': ',', 'explanation': 'backfield victoriouschoes reconc[,]', 'kl_divergence': 0.105414, 'mse': 1.511324, 'relative_rmse': 0.402581},
        {'position': 13, 'token': 'Manny', 'explanation': 'UFCournaments Neverthelessournaments[ Manny]', 'kl_divergence': 0.238154, 'mse': 1.235552, 'relative_rmse': 0.277266},
        {'position': 14, 'token': 'Gamb', 'explanation': 'Hopefully Manny Manny Miguel[ Gamb]', 'kl_divergence': 0.125416, 'mse': 1.062670, 'relative_rmse': 0.282262},
        {'position': 15, 'token': 'ury', 'explanation': 'Robb Gamb Gab Gamb[ury]', 'kl_divergence': 0.007880, 'mse': 1.009146, 'relative_rmse': 0.351258},
        {'position': 16, 'token': 'an', 'explanation': 'aminer Bryantmaxwellulia[an]', 'kl_divergence': 0.255645, 'mse': 1.402232, 'relative_rmse': 0.389238},
        {'position': 17, 'token': 'will', 'explanation': 'jandro Cabrera Ryuotto[ will]', 'kl_divergence': 0.408061, 'mse': 1.220898, 'relative_rmse': 0.366395},
        {'position': 18, 'token': 'return', 'explanation': 'Conorwill superstarwill[ return]', 'kl_divergence': 0.062677, 'mse': 1.083119, 'relative_rmse': 0.317893},
        {'position': 19, 'token': 'to', 'explanation': 'defenseman returnumerable returns[ to]', 'kl_divergence': 0.156334, 'mse': 0.982957, 'relative_rmse': 0.330180},
        {'position': 20, 'token': 'the', 'explanation': 'teammate returns to reborn[ the]', 'kl_divergence': 0.174606, 'mse': 0.775095, 'relative_rmse': 0.312729},
        {'position': 21, 'token': 'UFC', 'explanation': 'MMA adapting poised returning[ UFC]', 'kl_divergence': 1.233300, 'mse': 0.990372, 'relative_rmse': 0.259415},
        {'position': 22, 'token': 'oct', 'explanation': 'UFC UFC competing MMA[ oct]', 'kl_divergence': 0.007089, 'mse': 0.968365, 'relative_rmse': 0.270275},
        {'position': 23, 'token': 'agon', 'explanation': 'UFC boxing retire UFC[agon]', 'kl_divergence': 0.404347, 'mse': 1.713690, 'relative_rmse': 0.372360},
        {'position': 24, 'token': 'on', 'explanation': 'UFC TBA UFC rebirth[ on]', 'kl_divergence': 0.105190, 'mse': 0.943329, 'relative_rmse': 0.322389},
        {'position': 25, 'token': 'Sept', 'explanation': 'UFC premie on debut[ Sept]', 'kl_divergence': 0.001106, 'mse': 0.929879, 'relative_rmse': 0.288762},
        {'position': 26, 'token': '.', 'explanation': 'UFC Sept season Sept[.]', 'kl_divergence': 0.023824, 'mse': 0.699591, 'relative_rmse': 0.283873},
        {'position': 27, 'token': '27', 'explanation': 'UFC TBA January stage[ 27]', 'kl_divergence': 0.604515, 'mse': 1.096293, 'relative_rmse': 0.324756},
        {'position': 28, 'token': 'in', 'explanation': 'UFC kickoff Wednesday UFC[ in]', 'kl_divergence': 0.163225, 'mse': 0.714647, 'relative_rmse': 0.293818},
        {'position': 29, 'token': 'Las', 'explanation': 'UFC events outside marathon[ Las]', 'kl_divergence': 0.000127, 'mse': 0.698021, 'relative_rmse': 0.231145},
        {'position': 30, 'token': 'Vegas', 'explanation': 'UFC Seoul abroad Las[ Vegas]', 'kl_divergence': 0.899189, 'mse': 0.871904, 'relative_rmse': 0.255154},
        {'position': 31, 'token': 'at', 'explanation': 'UFC Vegas Anaheim Vegas[ at]', 'kl_divergence': 0.853140, 'mse': 0.792853, 'relative_rmse': 0.316090},
        {'position': 32, 'token': 'UFC', 'explanation': 'UFC fights atevent[ UFC]', 'kl_divergence': 0.066453, 'mse': 0.771785, 'relative_rmse': 0.249049},
        {'position': 33, 'token': '178', 'explanation': 'UFC UFC UFC at[ 178]', 'kl_divergence': 0.556908, 'mse': 0.704791, 'relative_rmse': 0.243800},
        {'position': 34, 'token': 'in', 'explanation': 'UFC UFC tournament TBD[ in]', 'kl_divergence': 0.330658, 'mse': 1.094007, 'relative_rmse': 0.355406},
        {'position': 35, 'token': 'a', 'explanation': 'UFC inside UFC abroad[ a]', 'kl_divergence': 0.245241, 'mse': 0.798317, 'relative_rmse': 0.312829},
        {'position': 36, 'token': 'new', 'explanation': 'UFC sanctioned inside scheduled[ new]', 'kl_divergence': 0.673917, 'mse': 0.785481, 'relative_rmse': 0.292964},
        {'position': 37, 'token': 'weight', 'explanation': 'UFC competed other upcoming[ weight]', 'kl_divergence': 0.054468, 'mse': 0.946437, 'relative_rmse': 0.286585},
        {'position': 38, 'token': 'class', 'explanation': 'revamped UFC new gym[ class]', 'kl_divergence': 0.364921, 'mse': 2.050936, 'relative_rmse': 0.424370},
        {'position': 39, 'token': '.', 'explanation': 'UFC contender MMA roster[.]', 'kl_divergence': 0.156551, 'mse': 0.523826, 'relative_rmse': 0.269099},
        {'position': 40, 'token': '\\n', 'explanation': 'NFL Additionally superstarEarlier[\\n]', 'kl_divergence': 0.000020, 'mse': 0.601572, 'relative_rmse': 0.295958},
        {'position': 41, 'token': '\\n', 'explanation': 'UFC event athlete debut[\\n]', 'kl_divergence': 0.039773, 'mse': 0.364208, 'relative_rmse': 0.213695},
        {'position': 42, 'token': 'G', 'explanation': 'Celticsicipated AdditionallyAsked[G]', 'kl_divergence': 0.031035, 'mse': 0.543566, 'relative_rmse': 0.212891},
        {'position': 43, 'token': 'amb', 'explanation': 'AlthoughGAlthoughG[amb]', 'kl_divergence': 0.001575, 'mse': 1.636334, 'relative_rmse': 0.331917},
        {'position': 44, 'token': 'ury', 'explanation': 'However MangGarKal[ury]', 'kl_divergence': 0.000660, 'mse': 1.301819, 'relative_rmse': 0.339183},
        {'position': 45, 'token': 'an', 'explanation': 'After McGregormaxwellJackson[an]', 'kl_divergence': 0.028599, 'mse': 0.930646, 'relative_rmse': 0.308827},
        {'position': 46, 'token': 'is', 'explanation': 'UFC AlonsoAlthough Diaz[ is]', 'kl_divergence': 0.124426, 'mse': 0.370572, 'relative_rmse': 0.217977},
        {'position': 47, 'token': 'set', 'explanation': 'defensemanhas Rouse is[ set]', 'kl_divergence': 0.041322, 'mse': 0.589685, 'relative_rmse': 0.253802},
        {'position': 48, 'token': 'to', 'explanation': 'UFC slatedis slated[ to]', 'kl_divergence': 0.155661, 'mse': 0.873735, 'relative_rmse': 0.322952},
        {'position': 49, 'token': 'make', 'explanation': 'UFC slated willUFC[ make]', 'kl_divergence': 0.040402, 'mse': 0.525270, 'relative_rmse': 0.252721},
        {'position': 50, 'token': 'his', 'explanation': 'midfielder makemakes earn[ his]', 'kl_divergence': 0.055562, 'mse': 0.808910, 'relative_rmse': 0.299233},
        {'position': 51, 'token': '135', 'explanation': 'UFC competed preparingWant[ 135]', 'kl_divergence': 0.043412, 'mse': 0.746384, 'relative_rmse': 0.265882},
        {'position': 52, 'token': '-', 'explanation': 'UFC 135 weigh 145[-]', 'kl_divergence': 0.001058, 'mse': 0.916245, 'relative_rmse': 0.294457},
        {'position': 53, 'token': 'pound', 'explanation': 'MMA competed 135 pounds[pound]', 'kl_divergence': 4.659106, 'mse': 1.237061, 'relative_rmse': 0.333183},
        {'position': 54, 'token': 'men', 'explanation': 'UFC crowned MMA 1[ men]', 'kl_divergence': 0.042226, 'mse': 1.273281, 'relative_rmse': 0.378677},
        {'position': 55, 'token': "'s", 'explanation': "NASCAR men mus men['s]", 'kl_divergence': 0.488049, 'mse': 1.322738, 'relative_rmse': 0.348014},
        {'position': 56, 'token': 'b', 'explanation': 'Lakers deserving MVP racing[ b]', 'kl_divergence': 0.006260, 'mse': 0.779769, 'relative_rmse': 0.277667},
        {'position': 57, 'token': 'antam', 'explanation': 'UFC b competed b[antam]', 'kl_divergence': 0.000182, 'mse': 0.651179, 'relative_rmse': 0.221559},
        {'position': 58, 'token': 'weight', 'explanation': 'UFC heavyweight boxingirm[weight]', 'kl_divergence': 4.857563, 'mse': 1.447081, 'relative_rmse': 0.352448},
        {'position': 59, 'token': 'debut', 'explanation': 'UFC heavyweight pitcher enter[ debut]', 'kl_divergence': 0.120006, 'mse': 1.234643, 'relative_rmse': 0.324171},
        {'position': 60, 'token': 'against', 'explanation': 'UFC debut boxer matchup[ against]', 'kl_divergence': 0.030874, 'mse': 0.528199, 'relative_rmse': 0.249494},
        {'position': 61, 'token': 'Cody', 'explanation': 'UFC matches fighters to[ Cody]', 'kl_divergence': 0.190434, 'mse': 0.263473, 'relative_rmse': 0.152115},
        {'position': 62, 'token': '"', 'explanation': 'UFC Cody Cody Cody[ "]', 'kl_divergence': 0.117426, 'mse': 0.507620, 'relative_rmse': 0.223089},
        {'position': 63, 'token': 'The', 'explanation': 'UFC punches like Jessica[The]', 'kl_divergence': 0.238618, 'mse': 1.037017, 'relative_rmse': 0.327670},
        {'position': 64, 'token': 'Reneg', 'explanation': 'fucking calling Being called[ Reneg]', 'kl_divergence': 0.007300, 'mse': 0.770244, 'relative_rmse': 0.245862},
        {'position': 65, 'token': 'ade', 'explanation': 'fighting Reneg Reneg Reneg[ade]', 'kl_divergence': 6.582366, 'mse': 1.622642, 'relative_rmse': 0.430452},
        {'position': 66, 'token': '"', 'explanation': 'UFC Curtis Conor Shane["]', 'kl_divergence': 0.106161, 'mse': 1.106041, 'relative_rmse': 0.349796},
        {'position': 67, 'token': 'Gibson', 'explanation': 'boxer Zimmerman catcher Hernandez[ Gibson]', 'kl_divergence': 0.060697, 'mse': 0.950746, 'relative_rmse': 0.270601},
        {'position': 68, 'token': 'at', 'explanation': 'UFC boxer matchup Diaz[ at]', 'kl_divergence': 0.023021, 'mse': 0.626333, 'relative_rmse': 0.265710},
        {'position': 69, 'token': 'the', 'explanation': 'UFC matchup at at[ the]', 'kl_divergence': 0.178307, 'mse': 0.605248, 'relative_rmse': 0.274840},
        {'position': 70, 'token': 'MGM', 'explanation': 'UFC attending at attending[ MGM]', 'kl_divergence': 0.001389, 'mse': 0.336722, 'relative_rmse': 0.150693},
        {'position': 71, 'token': 'Grand', 'explanation': 'concert MGM MGM MI[ Grand]', 'kl_divergence': 0.062327, 'mse': 1.180978, 'relative_rmse': 0.308666},
        {'position': 72, 'token': 'Arena', 'explanation': 'HBO Stadium Paramount San[ Arena]', 'kl_divergence': 0.099158, 'mse': 1.304176, 'relative_rmse': 0.316010},
        {'position': 73, 'token': 'on', 'explanation': 'UFC venue Cincinnati Arena[ on]', 'kl_divergence': 0.142350, 'mse': 0.482959, 'relative_rmse': 0.237004},
        {'position': 74, 'token': 'a', 'explanation': 'UFC session on MMA[ a]', 'kl_divergence': 0.264757, 'mse': 0.569112, 'relative_rmse': 0.273386},
        {'position': 75, 'token': 'pay', 'explanation': 'UFC scheduled on televised[ pay]', 'kl_divergence': 0.007656, 'mse': 0.494878, 'relative_rmse': 0.215882},
        {'position': 76, 'token': '-', 'explanation': 'UFC pay paid pay[-]', 'kl_divergence': 0.000029, 'mse': 0.665993, 'relative_rmse': 0.230387},
        {'position': 77, 'token': 'per', 'explanation': 'Netflix pay in pay[per]', 'kl_divergence': 0.000699, 'mse': 0.859305, 'relative_rmse': 0.249687},
        {'position': 78, 'token': '-', 'explanation': 'movieperper per[-]', 'kl_divergence': 0.001902, 'mse': 4.184834, 'relative_rmse': 0.558961},
        {'position': 79, 'token': 'view', 'explanation': 'UFC pre cash price[view]', 'kl_divergence': 0.868086, 'mse': 2.459902, 'relative_rmse': 0.483009},
        {'position': 80, 'token': 'card', 'explanation': 'UFC televised MMA scheduled[ card]', 'kl_divergence': 0.115015, 'mse': 1.249492, 'relative_rmse': 0.324263},
        {'position': 81, 'token': 'headlined', 'explanation': 'UFC event tournament event[ headlined]', 'kl_divergence': 0.000979, 'mse': 0.784043, 'relative_rmse': 0.263576},
        {'position': 82, 'token': 'by', 'explanation': 'UFC scheduled boxinghost[ by]', 'kl_divergence': 0.265073, 'mse': 1.569489, 'relative_rmse': 0.437305},
        {'position': 83, 'token': 'the', 'explanation': 'championship featuring commercials featuring[ the]', 'kl_divergence': 0.063994, 'mse': 0.739762, 'relative_rmse': 0.343558},
        {'position': 84, 'token': 'UFC', 'explanation': 'UFC promoting onring[ UFC]', 'kl_divergence': 0.434366, 'mse': 0.870501, 'relative_rmse': 0.272433},
        {'position': 85, 'token': 'light', 'explanation': 'UFC UFC sponsoring UFC[ light]', 'kl_divergence': 0.013711, 'mse': 0.622209, 'relative_rmse': 0.232872},
        {'position': 86, 'token': 'heavyweight', 'explanation': 'UFC wrestling heavy MMA[ heavyweight]', 'kl_divergence': 0.515532, 'mse': 1.280509, 'relative_rmse': 0.340865},
        {'position': 87, 'token': 'title', 'explanation': 'UFC boxing heavyweight with[ title]', 'kl_divergence': 0.162452, 'mse': 0.898369, 'relative_rmse': 0.277170},
        {'position': 88, 'token': 'fight', 'explanation': 'UFC boxing tournament title[ fight]', 'kl_divergence': 0.471735, 'mse': 1.357185, 'relative_rmse': 0.326427},
        {'position': 89, 'token': 'between', 'explanation': 'UFC fight MMA fight[ between]', 'kl_divergence': 0.054152, 'mse': 0.811835, 'relative_rmse': 0.292303},
        {'position': 90, 'token': 'champion', 'explanation': 'UFC fights betweenUFC[ champion]', 'kl_divergence': 0.067968, 'mse': 0.741215, 'relative_rmse': 0.256373},
        {'position': 91, 'token': 'Jon', 'explanation': 'UFC winner race champion[ Jon]', 'kl_divergence': 0.021418, 'mse': 0.589472, 'relative_rmse': 0.222716},
        {'position': 92, 'token': 'Jones', 'explanation': 'UFC fighters Jon Thor[ Jones]', 'kl_divergence': 0.151527, 'mse': 0.862978, 'relative_rmse': 0.266983},
        {'position': 93, 'token': 'and', 'explanation': 'UFC matchups boxer Jones[ and]', 'kl_divergence': 0.292615, 'mse': 1.061691, 'relative_rmse': 0.356606},
        {'position': 94, 'token': 'Daniel', 'explanation': 'UFC fighters because wrestler[ Daniel]', 'kl_divergence': 0.003832, 'mse': 0.542431, 'relative_rmse': 0.219433},
        {'position': 95, 'token': 'Corm', 'explanation': 'UFC Daniel Daniel if[ Corm]', 'kl_divergence': 0.000110, 'mse': 0.573455, 'relative_rmse': 0.203871},
        {'position': 96, 'token': 'ier', 'explanation': 'boxing Corm Mayweather if[ier]', 'kl_divergence': 0.005283, 'mse': 1.497045, 'relative_rmse': 0.365839},
        {'position': 97, 'token': 'r', 'explanation': 'baseball Morrison Cruzler[r]', 'kl_divergence': 0.394429, 'mse': 1.396997, 'relative_rmse': 0.425623}
    ]
}

if __name__ == '__main__':
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Generate Gamburyan with UFC visualization
    output_path = output_dir / 'gamburyan_with_ufc_compressed.svg'
    create_transcript_visualization(gamburyan_with_ufc_data, str(output_path))