#!/usr/bin/env python3
"""
Generate compressed Israel transcript visualization
"""

import sys
sys.path.append('.')
from generate_compressed_viz import create_transcript_visualization
from pathlib import Path

# Israel data
israel_data = {
    'original_text': "Israel Accused of Suppressing Terror Evidence to Help Out New Pal China\n\nIsrael is a country desperate for friends. Isolated in the Middle East and hated in large parts of the Arab world, it struggles to make alliances. The few it has, it guards fiercely. So it should perhaps come as no surprise that for years Israel has been courting China, inking trade deals and fêting one another over champagne. But that process now finds Israel in an awkward bind",
    'total_tokens': 96,
    'tokens': [
        {'position': 0, 'token': 'Israel', 'explanation': 'Referred oun vacated Ging[Israel]', 'kl_divergence': 8.896482, 'mse': 11591.706055, 'relative_rmse': 0.985106},
        {'position': 1, 'token': 'Acc', 'explanation': 'Traditionyip USSR Mexicans[ Acc]', 'kl_divergence': 0.170299, 'mse': 2.269813, 'relative_rmse': 0.442931},
        {'position': 2, 'token': 'used', 'explanation': 'quished Corruption Penal Penalty[used]', 'kl_divergence': 2.451023, 'mse': 3.797808, 'relative_rmse': 0.542153},
        {'position': 3, 'token': 'of', 'explanation': 'Genocide charged Guilty accused[ of]', 'kl_divergence': 5.948798, 'mse': 2.591908, 'relative_rmse': 0.442567},
        {'position': 4, 'token': 'Supp', 'explanation': 'Crimes accus massacres Crimes[ Supp]', 'kl_divergence': 1.259020, 'mse': 1.658899, 'relative_rmse': 0.351695},
        {'position': 5, 'token': 'ressing', 'explanation': 'Jews complicit Supp transgress[ressing]', 'kl_divergence': 3.152360, 'mse': 2.015057, 'relative_rmse': 0.408536},
        {'position': 6, 'token': 'Terror', 'explanation': 'Terrorism suppress Crus abuse[ Terror]', 'kl_divergence': 0.526522, 'mse': 2.041057, 'relative_rmse': 0.420526},
        {'position': 7, 'token': 'Evidence', 'explanation': 'Genocide genocide Genocide Genocide[ Evidence]', 'kl_divergence': 1.186830, 'mse': 2.449793, 'relative_rmse': 0.408173},
        {'position': 8, 'token': 'to', 'explanation': 'BDS pornography Abuse metadata[ to]', 'kl_divergence': 4.073861, 'mse': 1.887179, 'relative_rmse': 0.460458},
        {'position': 9, 'token': 'Help', 'explanation': 'genocide crimes documents to[ Help]', 'kl_divergence': 0.645968, 'mse': 2.023367, 'relative_rmse': 0.394658},
        {'position': 10, 'token': 'Out', 'explanation': 'oppression help help help[ Out]', 'kl_divergence': 0.898127, 'mse': 1.588253, 'relative_rmse': 0.410020},
        {'position': 11, 'token': 'New', 'explanation': 'terrorism expulsion Zionism Protect[ New]', 'kl_divergence': 2.563097, 'mse': 2.170086, 'relative_rmse': 0.450498},
        {'position': 12, 'token': 'Pal', 'explanation': 'terrorists new Genocide Updated[ Pal]', 'kl_divergence': 0.850332, 'mse': 1.303294, 'relative_rmse': 0.336603},
        {'position': 13, 'token': 'China', 'explanation': 'dystop Pal HUGE Monkey[ China]', 'kl_divergence': 1.991606, 'mse': 2.320415, 'relative_rmse': 0.434663},
        {'position': 14, 'token': '\\n', 'explanation': 'Shiva Zika Genocide Subway[\\n]', 'kl_divergence': 0.004338, 'mse': 1.209813, 'relative_rmse': 0.352850},
        {'position': 15, 'token': '\\n', 'explanation': 'BDS meme slideshow ransomware[\\n]', 'kl_divergence': 0.162655, 'mse': 1.422146, 'relative_rmse': 0.403791},
        {'position': 16, 'token': 'Israel', 'explanation': 'Campaign FurthermoreCritics Says[Israel]', 'kl_divergence': 0.285562, 'mse': 1.676767, 'relative_rmse': 0.341969},
        {'position': 17, 'token': 'is', 'explanation': 'Appears AzerbaijanPakistan UAE[ is]', 'kl_divergence': 0.478260, 'mse': 0.894238, 'relative_rmse': 0.306154},
        {'position': 18, 'token': 'a', 'explanation': 'Notwithstanding VenezuelaIs AMERICA[ a]', 'kl_divergence': 0.376399, 'mse': 0.846548, 'relative_rmse': 0.309220},
        {'position': 19, 'token': 'country', 'explanation': 'constitutedIran constitutedhas[ country]', 'kl_divergence': 0.060302, 'mse': 0.985088, 'relative_rmse': 0.286625},
        {'position': 20, 'token': 'desperate', 'explanation': 'quartered kingdom humankind Aman[ desperate]', 'kl_divergence': 0.127151, 'mse': 1.646590, 'relative_rmse': 0.345372},
        {'position': 21, 'token': 'for', 'explanation': 'Shiite desperate eager desperate[ for]', 'kl_divergence': 0.315922, 'mse': 1.397959, 'relative_rmse': 0.337462},
        {'position': 22, 'token': 'friends', 'explanation': 'Okinawa seek opportunities for[ friends]', 'kl_divergence': 0.258063, 'mse': 1.434826, 'relative_rmse': 0.345533},
        {'position': 23, 'token': '.', 'explanation': '龍喚士 adventurers exalted NGOs[.]', 'kl_divergence': 0.208076, 'mse': 0.968335, 'relative_rmse': 0.335385},
        {'position': 24, 'token': 'Is', 'explanation': 'imperialist Consequently! Resistance[ Is]', 'kl_divergence': 0.129947, 'mse': 0.767556, 'relative_rmse': 0.291278},
        {'position': 25, 'token': 'olated', 'explanation': 'Whereas dysfunctional Inolated[olated]', 'kl_divergence': 0.515996, 'mse': 1.825189, 'relative_rmse': 0.391277},
        {'position': 26, 'token': 'in', 'explanation': 'Confederacy afflicted Abandonolated[ in]', 'kl_divergence': 0.850828, 'mse': 1.286079, 'relative_rmse': 0.376589},
        {'position': 27, 'token': 'the', 'explanation': 'Exile isolated abroad isolated[ the]', 'kl_divergence': 0.199809, 'mse': 0.886702, 'relative_rmse': 0.320722},
        {'position': 28, 'token': 'Middle', 'explanation': 'Euras occupying abroad drifting[ Middle]', 'kl_divergence': 0.005771, 'mse': 1.203308, 'relative_rmse': 0.323059},
        {'position': 29, 'token': 'East', 'explanation': 'Thinking Europe Middle stagnation[ East]', 'kl_divergence': 0.395776, 'mse': 1.370629, 'relative_rmse': 0.327726},
        {'position': 30, 'token': 'and', 'explanation': 'Outs Europe Appalach Appalach[ and]', 'kl_divergence': 1.235092, 'mse': 1.602200, 'relative_rmse': 0.400268},
        {'position': 31, 'token': 'hated', 'explanation': 'empires because despisedcolonial[ hated]', 'kl_divergence': 2.994020, 'mse': 3.096009, 'relative_rmse': 0.497054},
        {'position': 32, 'token': 'in', 'explanation': 'culture disliked despised except[ in]', 'kl_divergence': 0.459110, 'mse': 1.394731, 'relative_rmse': 0.408545},
        {'position': 33, 'token': 'large', 'explanation': 'culture abroad insults Outside[ large]', 'kl_divergence': 0.581440, 'mse': 0.993541, 'relative_rmse': 0.332089},
        {'position': 34, 'token': 'parts', 'explanation': 'countries large inside rough[ parts]', 'kl_divergence': 0.102711, 'mse': 1.065795, 'relative_rmse': 0.350116},
        {'position': 35, 'token': 'of', 'explanation': 'countries turbulent corners mostly[ of]', 'kl_divergence': 1.428160, 'mse': 1.437318, 'relative_rmse': 0.421261},
        {'position': 36, 'token': 'the', 'explanation': 'rural parts fragmented mostly[ the]', 'kl_divergence': 0.454852, 'mse': 0.964353, 'relative_rmse': 0.334844},
        {'position': 37, 'token': 'Arab', 'explanation': 'societies dominating outside spreading[ Arab]', 'kl_divergence': 0.131066, 'mse': 0.775698, 'relative_rmse': 0.253751},
        {'position': 38, 'token': 'world', 'explanation': 'continent Arab strife Europe[ world]', 'kl_divergence': 1.111234, 'mse': 1.162389, 'relative_rmse': 0.307864},
        {'position': 39, 'token': ',', 'explanation': 'genreiterranean obscurityersion[,]', 'kl_divergence': 0.055258, 'mse': 1.098009, 'relative_rmse': 0.351114},
        {'position': 40, 'token': 'it', 'explanation': 'civilizationellationiancerieg[ it]', 'kl_divergence': 0.192295, 'mse': 1.092216, 'relative_rmse': 0.343132},
        {'position': 41, 'token': 'struggles', 'explanation': 'empiretheyigntyyet[ struggles]', 'kl_divergence': 0.086619, 'mse': 0.930354, 'relative_rmse': 0.278449},
        {'position': 42, 'token': 'to', 'explanation': 'civilisation struggling struggleforces[ to]', 'kl_divergence': 0.168601, 'mse': 0.874940, 'relative_rmse': 0.290451},
        {'position': 43, 'token': 'make', 'explanation': 'oppressed struggles difficult to[ make]', 'kl_divergence': 0.541779, 'mse': 0.864508, 'relative_rmse': 0.310036},
        {'position': 44, 'token': 'alliances', 'explanation': 'nations make precarious make[ alliances]', 'kl_divergence': 0.101433, 'mse': 0.814650, 'relative_rmse': 0.249089},
        {'position': 45, 'token': '.', 'explanation': 'empires strugglestrength alliances[.]', 'kl_divergence': 0.144791, 'mse': 0.719466, 'relative_rmse': 0.306370},
        {'position': 46, 'token': 'The', 'explanation': 'civilisation Nevertheless vitality Nevertheless[ The]', 'kl_divergence': 0.064179, 'mse': 0.332567, 'relative_rmse': 0.205769},
        {'position': 47, 'token': 'few', 'explanation': 'Socialist Making Having resentment[ few]', 'kl_divergence': 0.172278, 'mse': 0.622698, 'relative_rmse': 0.244772},
        {'position': 48, 'token': 'it', 'explanation': 'Returning handful Little peasants[ it]', 'kl_divergence': 0.739424, 'mse': 2.482628, 'relative_rmse': 0.498444},
        {'position': 49, 'token': 'has', 'explanation': 'Few they possessions they[ has]', 'kl_divergence': 0.482423, 'mse': 1.367727, 'relative_rmse': 0.374908},
        {'position': 50, 'token': ',', 'explanation': 'selves favored ones were[,]', 'kl_divergence': 0.639516, 'mse': 1.370497, 'relative_rmse': 0.428782},
        {'position': 51, 'token': 'it', 'explanation': 'aristilyites;[ it]', 'kl_divergence': 0.661002, 'mse': 1.413318, 'relative_rmse': 0.434782},
        {'position': 52, 'token': 'guards', 'explanation': 'empire nobodythey foe[ guards]', 'kl_divergence': 0.247496, 'mse': 0.937587, 'relative_rmse': 0.291072},
        {'position': 53, 'token': 'fiercely', 'explanation': 'Barbar guard guard keep[ fiercely]', 'kl_divergence': 0.119781, 'mse': 0.917999, 'relative_rmse': 0.265539},
        {'position': 54, 'token': '.', 'explanation': 'armies steadykeep steady[.]', 'kl_divergence': 0.128995, 'mse': 0.680448, 'relative_rmse': 0.302081},
        {'position': 55, 'token': 'So', 'explanation': 'turbulent Moreover strong aristocracy[ So]', 'kl_divergence': 0.124517, 'mse': 0.527358, 'relative_rmse': 0.247418},
        {'position': 56, 'token': 'it', 'explanation': 'civilizedSoThusso[ it]', 'kl_divergence': 0.784294, 'mse': 1.095583, 'relative_rmse': 0.354081},
        {'position': 57, 'token': 'should', 'explanation': 'majestic Itso it[ should]', 'kl_divergence': 0.410980, 'mse': 1.069581, 'relative_rmse': 0.351620},
        {'position': 58, 'token': 'perhaps', 'explanation': 'isSpecialOrderable itshould surely[ perhaps]', 'kl_divergence': 0.255834, 'mse': 1.235110, 'relative_rmse': 0.354082},
        {'position': 59, 'token': 'come', 'explanation': 'proverb might surely maybe[ come]', 'kl_divergence': 0.059708, 'mse': 1.178870, 'relative_rmse': 0.354955},
        {'position': 60, 'token': 'as', 'explanation': 'come come come It[ as]', 'kl_divergence': 0.068944, 'mse': 1.002735, 'relative_rmse': 0.336960},
        {'position': 61, 'token': 'no', 'explanation': 'come as come as[ no]', 'kl_divergence': 0.001765, 'mse': 1.078633, 'relative_rmse': 0.338911},
        {'position': 62, 'token': 'surprise', 'explanation': 'hardly shouldn no suffice[ surprise]', 'kl_divergence': 0.055758, 'mse': 1.087016, 'relative_rmse': 0.278971},
        {'position': 63, 'token': 'that', 'explanation': 'unsurintuitive unlikelyCome[ that]', 'kl_divergence': 0.127692, 'mse': 1.090716, 'relative_rmse': 0.354256},
        {'position': 64, 'token': 'for', 'explanation': 'nationalists Nonetheless irrational Reasons[ for]', 'kl_divergence': 1.808601, 'mse': 2.539876, 'relative_rmse': 0.554053},
        {'position': 65, 'token': 'years', 'explanation': 'Opportun for Opportun crises[ years]', 'kl_divergence': 0.416151, 'mse': 1.323780, 'relative_rmse': 0.317247},
        {'position': 66, 'token': 'Israel', 'explanation': 'selves revolutionsYearsstill[ Israel]', 'kl_divergence': 0.052572, 'mse': 1.148149, 'relative_rmse': 0.294045},
        {'position': 67, 'token': 'has', 'explanation': 'empirestoday Finlandtoday[ has]', 'kl_divergence': 0.129285, 'mse': 0.947769, 'relative_rmse': 0.324051},
        {'position': 68, 'token': 'been', 'explanation': 'empires latelyAmericansijing[ been]', 'kl_divergence': 0.136417, 'mse': 0.684329, 'relative_rmse': 0.264541},
        {'position': 69, 'token': 'cour', 'explanation': 'monarchy exists been administrations[ cour]', 'kl_divergence': 0.010639, 'mse': 0.415358, 'relative_rmse': 0.189296},
        {'position': 70, 'token': 'ting', 'explanation': 'empires pursued cour flourished[ting]', 'kl_divergence': 0.066958, 'mse': 0.923434, 'relative_rmse': 0.282101},
        {'position': 71, 'token': 'China', 'explanation': 'hegemony befriend negotiate attracts[ China]', 'kl_divergence': 0.273369, 'mse': 0.822548, 'relative_rmse': 0.248193},
        {'position': 72, 'token': ',', 'explanation': 'libertarians greed Brazil lured[,]', 'kl_divergence': 0.083846, 'mse': 0.975129, 'relative_rmse': 0.340778},
        {'position': 73, 'token': 'in', 'explanation': "empires', libertarians),[ in]", 'kl_divergence': 0.097189, 'mse': 0.833100, 'relative_rmse': 0.306983},
        {'position': 74, 'token': 'king', 'explanation': 'empires abroad, negoti[king]', 'kl_divergence': 0.154298, 'mse': 1.137801, 'relative_rmse': 0.345018},
        {'position': 75, 'token': 'trade', 'explanation': 'Euras sign forge sign[ trade]', 'kl_divergence': 0.025879, 'mse': 1.027029, 'relative_rmse': 0.281549},
        {'position': 76, 'token': 'deals', 'explanation': 'bourgeois sell tradeexpl[ deals]', 'kl_divergence': 0.057988, 'mse': 0.917691, 'relative_rmse': 0.267867},
        {'position': 77, 'token': 'and', 'explanation': 'Industry bribes trade tricks[ and]', 'kl_divergence': 0.328968, 'mse': 1.076945, 'relative_rmse': 0.351173},
        {'position': 78, 'token': 'f', 'explanation': 'capitalist because exports because[ f]', 'kl_divergence': 0.095411, 'mse': 0.534448, 'relative_rmse': 0.217784},
        {'position': 79, 'token': 'ê', 'explanation': 'Chinese f f f[ê]', 'kl_divergence': 0.484586, 'mse': 0.630891, 'relative_rmse': 0.232488},
        {'position': 80, 'token': 'ting', 'explanation': 'diplomacy fe felike[ting]', 'kl_divergence': 1.153091, 'mse': 2.372235, 'relative_rmse': 0.498460},
        {'position': 81, 'token': 'one', 'explanation': 'capitalist sell but promote[ one]', 'kl_divergence': 0.409867, 'mse': 1.124277, 'relative_rmse': 0.381722},
        {'position': 82, 'token': 'another', 'explanation': 'trade mutual one swap[ another]', 'kl_divergence': 0.262643, 'mse': 1.011967, 'relative_rmse': 0.320176},
        {'position': 83, 'token': 'over', 'explanation': 'populism pact capitalists exchange[ over]', 'kl_divergence': 0.161645, 'mse': 0.884570, 'relative_rmse': 0.332901},
        {'position': 84, 'token': 'champagne', 'explanation': 'socialists talk on praise[ champagne]', 'kl_divergence': 0.083187, 'mse': 0.779294, 'relative_rmse': 0.255110},
        {'position': 85, 'token': '.', 'explanation': 'capitalists opium greed splendid[.]', 'kl_divergence': 0.161412, 'mse': 0.595960, 'relative_rmse': 0.285720},
        {'position': 86, 'token': 'But', 'explanation': 'populist flourish Moreover prosperity[ But]', 'kl_divergence': 0.136901, 'mse': 0.552618, 'relative_rmse': 0.238918},
        {'position': 87, 'token': 'that', 'explanation': 'But Yet Yet But[ that]', 'kl_divergence': 0.033616, 'mse': 0.601886, 'relative_rmse': 0.259981},
        {'position': 88, 'token': 'process', 'explanation': 'populism Such However But[ process]', 'kl_divergence': 0.060617, 'mse': 0.813262, 'relative_rmse': 0.265212},
        {'position': 89, 'token': 'now', 'explanation': 'However reasoning discourse This[ now]', 'kl_divergence': 0.212911, 'mse': 1.195868, 'relative_rmse': 0.327974},
        {'position': 90, 'token': 'finds', 'explanation': 'contradictionnow situationnow[ finds]', 'kl_divergence': 0.111425, 'mse': 0.751450, 'relative_rmse': 0.289258},
        {'position': 91, 'token': 'Israel', 'explanation': 'leadershipnow Thenow[ Israel]', 'kl_divergence': 0.066413, 'mse': 0.785379, 'relative_rmse': 0.255892},
        {'position': 92, 'token': 'in', 'explanation': 'crisis nation Israel itself[ in]', 'kl_divergence': 0.032361, 'mse': 0.761607, 'relative_rmse': 0.300226},
        {'position': 93, 'token': 'an', 'explanation': 'crisis in into in[ an]', 'kl_divergence': 0.066588, 'mse': 0.727514, 'relative_rmse': 0.297475},
        {'position': 94, 'token': 'awkward', 'explanation': 'crisis a an fragile[ awkward]', 'kl_divergence': 0.113536, 'mse': 0.794447, 'relative_rmse': 0.261173},
        {'position': 95, 'token': 'bind', 'explanation': 'positionawkward precarious dilemma[ bind]', 'kl_divergence': 0.150651, 'mse': 0.796116, 'relative_rmse': 0.239103}
    ]
}

if __name__ == '__main__':
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Generate Israel visualization
    output_path = output_dir / 'israel_compressed.svg'
    create_transcript_visualization(israel_data, str(output_path))