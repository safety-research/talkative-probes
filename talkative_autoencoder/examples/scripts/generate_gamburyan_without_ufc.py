#!/usr/bin/env python3
"""
Generate compressed Gamburyan (without UFC) transcript visualization
"""

import sys
sys.path.append('.')
from generate_compressed_viz import create_transcript_visualization
from pathlib import Path

# Gamburyan data without UFC (95 tokens)
gamburyan_without_ufc_data = {
    'original_text': 'Looking to leave a two-fight winless streak behind him, Manny Gamburyan will return to the octagon on Sept. 27 in Las Vegas at 178 in a new weight class.\n\nGamburyan is set to make his 135-pound men\'s bantamweight debut against Cody "The Renegade" Gibson at the MGM Grand Arena on a pay-per-view card headlined by the light heavyweight title fight between champion Jon Jones and Daniel Cormierr',
    'total_tokens': 95,
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
        {'position': 21, 'token': 'oct', 'explanation': 'UFC competing serving returning[ oct]', 'kl_divergence': 0.004549, 'mse': 0.924522, 'relative_rmse': 0.272414},
        {'position': 22, 'token': 'agon', 'explanation': 'MMA stepping retire Seoul[agon]', 'kl_divergence': 0.602924, 'mse': 1.548175, 'relative_rmse': 0.357336},
        {'position': 23, 'token': 'on', 'explanation': 'UFC comeback arena TBA[ on]', 'kl_divergence': 0.101774, 'mse': 0.945413, 'relative_rmse': 0.329811},
        {'position': 24, 'token': 'Sept', 'explanation': 'UFC TBA on matchup[ Sept]', 'kl_divergence': 0.002106, 'mse': 0.812603, 'relative_rmse': 0.273163},
        {'position': 25, 'token': '.', 'explanation': 'UFC Oct roster Sept[.]', 'kl_divergence': 0.125823, 'mse': 0.847231, 'relative_rmse': 0.312889},
        {'position': 26, 'token': '27', 'explanation': 'UFC TBA Sept schedule[ 27]', 'kl_divergence': 0.742986, 'mse': 0.976252, 'relative_rmse': 0.308280},
        {'position': 27, 'token': 'in', 'explanation': 'UFC kickoff January Tuesday[ in]', 'kl_divergence': 0.180030, 'mse': 0.800241, 'relative_rmse': 0.314309},
        {'position': 28, 'token': 'Las', 'explanation': 'UFC events outside finals[ Las]', 'kl_divergence': 0.000976, 'mse': 0.699024, 'relative_rmse': 0.232404},
        {'position': 29, 'token': 'Vegas', 'explanation': 'playoffs Las Vegas outside[ Vegas]', 'kl_divergence': 0.850089, 'mse': 0.907258, 'relative_rmse': 0.261451},
        {'position': 30, 'token': 'at', 'explanation': 'UFC Vegas Vegas tournament[ at]', 'kl_divergence': 0.556358, 'mse': 0.793397, 'relative_rmse': 0.316645},
        {'position': 31, 'token': '178', 'explanation': 'UFC bouts at Sunday[ 178]', 'kl_divergence': 0.833235, 'mse': 0.697060, 'relative_rmse': 0.274472},
        {'position': 32, 'token': 'in', 'explanation': 'match 208 MMA 135[ in]', 'kl_divergence': 1.298564, 'mse': 1.443215, 'relative_rmse': 0.454784},
        {'position': 33, 'token': 'a', 'explanation': 'UFC indoors 30 abroad[ a]', 'kl_divergence': 0.501265, 'mse': 0.822472, 'relative_rmse': 0.329872},
        {'position': 34, 'token': 'new', 'explanation': 'MMA sanctioned inside convened[ new]', 'kl_divergence': 0.383711, 'mse': 0.928410, 'relative_rmse': 0.325783},
        {'position': 35, 'token': 'weight', 'explanation': 'UFC promoted new MMA[ weight]', 'kl_divergence': 0.070944, 'mse': 0.766752, 'relative_rmse': 0.263375},
        {'position': 36, 'token': 'class', 'explanation': 'sanctioned MMA scaled new[ class]', 'kl_divergence': 0.298117, 'mse': 1.966395, 'relative_rmse': 0.417441},
        {'position': 37, 'token': '.', 'explanation': 'UFC winner lineup athlete[.]', 'kl_divergence': 0.063313, 'mse': 0.556334, 'relative_rmse': 0.279490},
        {'position': 38, 'token': '\\n', 'explanation': 'championshipAfter athleteTickets[\\n]', 'kl_divergence': 0.000182, 'mse': 0.551821, 'relative_rmse': 0.287929},
        {'position': 39, 'token': '\\n', 'explanation': 'UFC winner promotion athlete[\\n]', 'kl_divergence': 0.028813, 'mse': 0.378807, 'relative_rmse': 0.219161},
        {'position': 40, 'token': 'G', 'explanation': 'CelticsNonetheless AskedEarlier[G]', 'kl_divergence': 0.011468, 'mse': 0.415329, 'relative_rmse': 0.186368},
        {'position': 41, 'token': 'amb', 'explanation': 'HoweverGAlthoughG[amb]', 'kl_divergence': 0.000923, 'mse': 1.445821, 'relative_rmse': 0.315699},
        {'position': 42, 'token': 'ury', 'explanation': 'DespiteKalAmbKar[ury]', 'kl_divergence': 0.000694, 'mse': 1.257730, 'relative_rmse': 0.339437},
        {'position': 43, 'token': 'an', 'explanation': 'According McGregormaxwellAnderson[an]', 'kl_divergence': 0.033514, 'mse': 1.001079, 'relative_rmse': 0.322875},
        {'position': 44, 'token': 'is', 'explanation': 'During outfielderAnthony½[ is]', 'kl_divergence': 0.131411, 'mse': 0.554025, 'relative_rmse': 0.268110},
        {'position': 45, 'token': 'set', 'explanation': 'defensemanhas As Rouse[ set]', 'kl_divergence': 0.017672, 'mse': 0.804108, 'relative_rmse': 0.301198},
        {'position': 46, 'token': 'to', 'explanation': 'UFC scheduledWill slated[ to]', 'kl_divergence': 0.119935, 'mse': 0.868883, 'relative_rmse': 0.324927},
        {'position': 47, 'token': 'make', 'explanation': 'defenseman expects slated Ald[ make]', 'kl_divergence': 0.142757, 'mse': 0.576760, 'relative_rmse': 0.265879},
        {'position': 48, 'token': 'his', 'explanation': 'midfielder make prepares earn[ his]', 'kl_divergence': 0.088082, 'mse': 0.563460, 'relative_rmse': 0.250931},
        {'position': 49, 'token': '135', 'explanation': 'MMA competed becomingReady[ 135]', 'kl_divergence': 1.164477, 'mse': 1.045859, 'relative_rmse': 0.316634},
        {'position': 50, 'token': '-', 'explanation': 'LeBron 135 undefeated 155[-]', 'kl_divergence': 0.039001, 'mse': 0.877813, 'relative_rmse': 0.294416},
        {'position': 51, 'token': 'pound', 'explanation': 'MMA competed 135 than[pound]', 'kl_divergence': 4.205867, 'mse': 1.065459, 'relative_rmse': 0.315825},
        {'position': 52, 'token': 'men', 'explanation': 'UFC crowned lb make[ men]', 'kl_divergence': 0.071023, 'mse': 0.830125, 'relative_rmse': 0.314611},
        {'position': 53, 'token': "'s", 'explanation': "competitor men men mus['s]", 'kl_divergence': 0.991181, 'mse': 1.361116, 'relative_rmse': 0.356504},
        {'position': 54, 'token': 'b', 'explanation': 'MVP vying champion riding[ b]', 'kl_divergence': 0.013938, 'mse': 0.650462, 'relative_rmse': 0.256302},
        {'position': 55, 'token': 'antam', 'explanation': 'UFC married b wrestlers[antam]', 'kl_divergence': 0.000610, 'mse': 0.740269, 'relative_rmse': 0.244906},
        {'position': 56, 'token': 'weight', 'explanation': 'UFC heavyweight wrestle brun[weight]', 'kl_divergence': 7.589307, 'mse': 1.551933, 'relative_rmse': 0.369853},
        {'position': 57, 'token': 'debut', 'explanation': 'UFC heavyweight compete pitcher[ debut]', 'kl_divergence': 0.057043, 'mse': 1.120284, 'relative_rmse': 0.310823},
        {'position': 58, 'token': 'against', 'explanation': 'UFC heavyweight debut bout[ against]', 'kl_divergence': 0.030651, 'mse': 0.463825, 'relative_rmse': 0.233455},
        {'position': 59, 'token': 'Cody', 'explanation': 'UFC fighters matches to[ Cody]', 'kl_divergence': 0.642811, 'mse': 0.264439, 'relative_rmse': 0.154685},
        {'position': 60, 'token': '"', 'explanation': 'UFC Cody Cody Mark[ "]', 'kl_divergence': 0.108808, 'mse': 0.480172, 'relative_rmse': 0.219150},
        {'position': 61, 'token': 'The', 'explanation': 'WWE fighters named Muhammad[The]', 'kl_divergence': 0.176552, 'mse': 0.947101, 'relative_rmse': 0.318744},
        {'position': 62, 'token': 'Reneg', 'explanation': 'punk calling Being looking[ Reneg]', 'kl_divergence': 0.111666, 'mse': 0.671055, 'relative_rmse': 0.231081},
        {'position': 63, 'token': 'ade', 'explanation': 'fighting Reneg Reneg Reneg[ade]', 'kl_divergence': 6.426089, 'mse': 1.633935, 'relative_rmse': 0.431930},
        {'position': 64, 'token': '"', 'explanation': 'boxer Rafael Charlie Darrell["]', 'kl_divergence': 0.095098, 'mse': 1.093151, 'relative_rmse': 0.348061},
        {'position': 65, 'token': 'Gibson', 'explanation': 'boxer Medina catcher Gonzalez[ Gibson]', 'kl_divergence': 0.037910, 'mse': 0.711961, 'relative_rmse': 0.233802},
        {'position': 66, 'token': 'at', 'explanation': 'UFC boxer boxer Goku[ at]', 'kl_divergence': 0.047643, 'mse': 0.790838, 'relative_rmse': 0.298013},
        {'position': 67, 'token': 'the', 'explanation': 'UFC champ at showdown[ the]', 'kl_divergence': 0.048388, 'mse': 0.536436, 'relative_rmse': 0.258705},
        {'position': 68, 'token': 'MGM', 'explanation': 'UFC attending at touring[ MGM]', 'kl_divergence': 0.001728, 'mse': 0.334731, 'relative_rmse': 0.150130},
        {'position': 69, 'token': 'Grand', 'explanation': 'concert MGM MGM MI[ Grand]', 'kl_divergence': 0.020426, 'mse': 1.058472, 'relative_rmse': 0.295252},
        {'position': 70, 'token': 'Arena', 'explanation': 'HBO Stadium Arena Toronto[ Arena]', 'kl_divergence': 0.092139, 'mse': 1.323457, 'relative_rmse': 0.319761},
        {'position': 71, 'token': 'on', 'explanation': 'UFC venue Arena MMA[ on]', 'kl_divergence': 0.112146, 'mse': 0.540191, 'relative_rmse': 0.250131},
        {'position': 72, 'token': 'a', 'explanation': 'UFC on scheduled venue[ a]', 'kl_divergence': 0.264756, 'mse': 0.661867, 'relative_rmse': 0.296181},
        {'position': 73, 'token': 'pay', 'explanation': 'UFC hosted scheduled on[ pay]', 'kl_divergence': 0.002409, 'mse': 0.426797, 'relative_rmse': 0.202712},
        {'position': 74, 'token': '-', 'explanation': 'UFC pay paid pay[-]', 'kl_divergence': 0.000169, 'mse': 0.560222, 'relative_rmse': 0.213565},
        {'position': 75, 'token': 'per', 'explanation': 'PlayStation pay in pay[per]', 'kl_divergence': 0.000233, 'mse': 0.827951, 'relative_rmse': 0.246605},
        {'position': 76, 'token': '-', 'explanation': 'movieperper pay[-]', 'kl_divergence': 0.000275, 'mse': 2.385902, 'relative_rmse': 0.423624},
        {'position': 77, 'token': 'view', 'explanation': 'UFC promotional per paid[view]', 'kl_divergence': 0.487621, 'mse': 1.500405, 'relative_rmse': 0.378813},
        {'position': 78, 'token': 'card', 'explanation': 'UFC broadcast Sunday MMA[ card]', 'kl_divergence': 0.084257, 'mse': 1.578051, 'relative_rmse': 0.369061},
        {'position': 79, 'token': 'headlined', 'explanation': 'tournament broadcast tournament event[ headlined]', 'kl_divergence': 0.002043, 'mse': 0.904699, 'relative_rmse': 0.287347},
        {'position': 80, 'token': 'by', 'explanation': 'arena scheduled commercials headlined[ by]', 'kl_divergence': 0.416709, 'mse': 1.443401, 'relative_rmse': 0.417001},
        {'position': 81, 'token': 'the', 'explanation': 'arena commercials related with[ the]', 'kl_divergence': 0.111005, 'mse': 0.733900, 'relative_rmse': 0.342942},
        {'position': 82, 'token': 'light', 'explanation': 'UFC representing athletes representing[ light]', 'kl_divergence': 0.025630, 'mse': 0.558284, 'relative_rmse': 0.226355},
        {'position': 83, 'token': 'heavyweight', 'explanation': 'UFC cutting lightHeavy[ heavyweight]', 'kl_divergence': 0.050291, 'mse': 0.572853, 'relative_rmse': 0.228953},
        {'position': 84, 'token': 'title', 'explanation': 'UFC promoting heavyweight MMA[ title]', 'kl_divergence': 0.184834, 'mse': 0.932446, 'relative_rmse': 0.287285},
        {'position': 85, 'token': 'fight', 'explanation': 'UFC boxing championshiptitle[ fight]', 'kl_divergence': 0.327499, 'mse': 1.267405, 'relative_rmse': 0.313235},
        {'position': 86, 'token': 'between', 'explanation': 'UFC match boxing fight[ between]', 'kl_divergence': 0.090609, 'mse': 0.657417, 'relative_rmse': 0.264883},
        {'position': 87, 'token': 'champion', 'explanation': 'UFC fights between fighters[ champion]', 'kl_divergence': 0.062562, 'mse': 0.693550, 'relative_rmse': 0.259478},
        {'position': 88, 'token': 'Jon', 'explanation': 'UFC champions champion boxer[ Jon]', 'kl_divergence': 0.018768, 'mse': 0.566140, 'relative_rmse': 0.225690},
        {'position': 89, 'token': 'Jones', 'explanation': 'fighters UFC Jon riders[ Jones]', 'kl_divergence': 0.229175, 'mse': 0.863716, 'relative_rmse': 0.271263},
        {'position': 90, 'token': 'and', 'explanation': 'UFC matchup Mayweather Johnson[ and]', 'kl_divergence': 0.225848, 'mse': 0.976578, 'relative_rmse': 0.348186},
        {'position': 91, 'token': 'Daniel', 'explanation': 'UFC fights because wrestler[ Daniel]', 'kl_divergence': 0.062100, 'mse': 0.500387, 'relative_rmse': 0.214864},
        {'position': 92, 'token': 'Corm', 'explanation': 'UFC Daniel Daniel if[ Corm]', 'kl_divergence': 0.000058, 'mse': 0.535657, 'relative_rmse': 0.197563},
        {'position': 93, 'token': 'ier', 'explanation': 'boxing Corm Mayweather if[ier]', 'kl_divergence': 0.014170, 'mse': 1.385943, 'relative_rmse': 0.358528},
        {'position': 94, 'token': 'r', 'explanation': 'baseball Ramirezler Brewer[r]', 'kl_divergence': 0.524683, 'mse': 1.797087, 'relative_rmse': 0.482663}
    ]
}

if __name__ == '__main__':
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Generate Gamburyan without UFC visualization
    output_path = output_dir / 'gamburyan_without_ufc_compressed.svg'
    create_transcript_visualization(gamburyan_without_ufc_data, str(output_path))