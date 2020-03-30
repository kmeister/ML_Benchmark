//
// autogenerated weights for a 3 hidden layer MLP
// Model Generator Created by Kurt Meister, Model generated  2020-03-30 17:24:58.025205
//

#ifndef BENCHMARKS_WEIGHTS_H
#define BENCHMARKS_WEIGHTS_H// Dimensions and Weights (including biases in first col) for LAYER_1
const int LAYER_1_N_NEURONS = 10;
const int LAYER_1_N_WEIGHTS = 4;
float LAYER_1_WEIGHTS[10*4] = {
	 0.02801380306482315 , -0.6708608865737915 , 1.216963529586792 , -1.3276153802871704 ,
	 -0.019388310611248016 , 0.49822115898132324 , -0.7792037725448608 , -0.6271531581878662 ,
	 -0.09873940050601959 , 0.4606638252735138 , -0.10017011314630508 , 0.09955046325922012 ,
	 0.06794587522745132 , -1.391121745109558 , -0.630061149597168 , 0.05525285750627518 ,
	 0.07090893387794495 , -1.3037729263305664 , 0.6507489681243896 , 0.8099438548088074 ,
	 -0.058722466230392456 , 0.4902379810810089 , 1.1054353713989258 , 0.24783770740032196 ,
	 -0.13547493517398834 , 0.5929982662200928 , 0.24753877520561218 , -0.32875046133995056 ,
	 0.18537278473377228 , -1.1108002662658691 , -0.7264645099639893 , 0.8907805681228638 ,
	 0.08371488749980927 , -1.1296138763427734 , -0.0953604206442833 , -1.0422776937484741 ,
	 0.06561306864023209 , -1.3258237838745117 , -0.8304034471511841 , 0.2970883548259735 
};

// Dimensions and Weights (including biases in first col) for LAYER_2
const int LAYER_2_N_NEURONS = 5;
const int LAYER_2_N_WEIGHTS = 11;
float LAYER_2_WEIGHTS[5*11] = {
	 -0.21076056361198425 , -0.20357859134674072 , 0.0859055295586586 , -0.06650083512067795 , -0.6734882593154907 , 0.27236825227737427 , 0.479704886674881 , -0.21953268349170685 , -0.2719857096672058 , -0.426212877035141 , -0.24736441671848297 ,
	 0.08697520196437836 , -0.04459823668003082 , 0.7650647163391113 , 0.2648923397064209 , -0.3013775944709778 , 0.1475939303636551 , 1.0043632984161377 , 0.4450441598892212 , -0.4168183207511902 , -0.005387242883443832 , 0.7254214882850647 ,
	 -0.1901407092809677 , -0.07232696563005447 , -0.3840814530849457 , 0.5240163803100586 , 0.20558783411979675 , 0.9619102478027344 , 0.10140951722860336 , -0.08283315598964691 , 0.7820308208465576 , 0.5657083988189697 , 0.5560539960861206 ,
	 0.09577419608831406 , 0.3635861575603485 , 0.6924931406974792 , 0.5367213487625122 , 0.2713043987751007 , 0.590888261795044 , 0.5172200798988342 , -0.5271680951118469 , -0.6211423277854919 , -0.7495200037956238 , 0.08393130451440811 ,
	 0.07133738696575165 , 0.3953689634799957 , 0.18013788759708405 , 0.2587454915046692 , 0.11754728853702545 , -0.6202653646469116 , -0.6378849744796753 , -0.09155680239200592 , 0.1564950793981552 , 0.24803197383880615 , 0.56313556432724 
};

// Dimensions and Weights (including biases in first col) for LAYER_3
const int LAYER_3_N_NEURONS = 2;
const int LAYER_3_N_WEIGHTS = 6;
float LAYER_3_WEIGHTS[2*6] = {
	 0.28441500663757324 , -0.8055289387702942 , -0.08440068364143372 , -0.9653444290161133 , 0.34297412633895874 , -0.6835673451423645 ,
	 -0.28441500663757324 , -0.3099197745323181 , -1.3204991817474365 , -0.007707458920776844 , -0.3986469507217407 , 0.09068591147661209 
};

// Array containing TEST  input values for the ANN
const int TEST_N_INPUTS = 250;
const int TEST_N_FEATURES = 3;
float TEST_INPUTS[250*3] = {
	0.3362253931021546 , 0.6752542742881273 , 0.3664788636184124 ,
	0.5304099139807354 , -1.695822251286762 , 0.8402718369239617 ,
	-0.022169632260188532 , 0.31335826008151824 , 1.295793663523256 ,
	-1.339995443559555 , -1.0549695662079794 , 0.5450148472463461 ,
	0.5716258542557393 , 0.9243048293851555 , -0.4520444562580872 ,
	-1.6595922184974277 , -2.107740470926988 , 1.0007890975565836 ,
	1.2926086372950945 , -1.315677985954717 , -0.1296311724250112 ,
	1.1851537918013657 , -1.4034570061027603 , 0.0694102866455466 ,
	0.4216940000404381 , 1.131292734623479 , -0.5580896497006893 ,
	0.004552157928013778 , 0.26126198313265814 , 1.4644277567558328 ,
	1.141359342277188 , 0.48620030537032377 , -0.28730515325142425 ,
	1.4423477098223134 , 1.8801630914970122 , -3.02587955054983 ,
	1.1488635630735473 , 1.119757691975326 , -1.4883099481025108 ,
	-0.7881725895607322 , 0.3708954511653402 , 1.2008130646679602 ,
	0.8700596671653035 , 0.09101673765927432 , -0.7291829721588308 ,
	-0.0998385001639126 , 0.8064873759072462 , 2.221242629371954 ,
	-3.299644924165007 , 3.188803963369419 , -2.051312187581103 ,
	0.5041965538042734 , -0.9868523233703949 , -0.8463413831152843 ,
	-0.5169450056683145 , -0.20977119710293823 , 1.9993922997539864 ,
	-0.022333466431151194 , 0.3065802589260722 , -1.1842314996821532 ,
	0.9853741024308735 , -0.07460973681931116 , -2.648914543191789 ,
	0.25455369174050224 , 0.42161438348102087 , 0.9145972786452234 ,
	0.05518762423615042 , 0.6670160308541464 , -1.74111806095679 ,
	0.4170018322313408 , 0.6826608677626289 , -3.2002677577757397 ,
	-1.3071611644920849 , 1.2538769367411715 , -1.8341835694537698 ,
	-1.0418978924915243 , -0.5773011903466202 , 1.1096317143378198 ,
	1.4972275346955968 , -1.892364736394836 , -2.304311128164854 ,
	0.4311646545561362 , 0.8108398090766848 , -0.15225328058538445 ,
	-1.3172122264690218 , 1.2176285324271137 , -0.7045453401466485 ,
	-0.057843210046011784 , 0.15720554806607412 , -2.413961023031602 ,
	-1.8780856356095113 , -3.450663855516233 , -0.3921041812392836 ,
	0.42944716397960014 , -0.21933113581098906 , -2.452823112704618 ,
	-1.0343875086090775 , -1.0026018667617094 , 1.2838751890616609 ,
	-1.2153923057973433 , -2.2111953338318244 , 0.6886723767279799 ,
	0.7763671016333402 , -0.4699581033201473 , -1.6957081399527565 ,
	0.4903293074887052 , 0.17327259597382483 , 0.914870571577062 ,
	-1.2846023887602716 , 0.8905589622337351 , 0.4411873690360921 ,
	-0.8030247405469199 , 0.037154284758104894 , 1.9137002476003344 ,
	0.3917018733979396 , -0.18965355900966707 , 0.6676576618552881 ,
	1.7978541153772891 , -0.09910994648320437 , -1.7674889295383636 ,
	-0.4386678120887416 , -1.471609840382071 , 1.7767954941031312 ,
	-1.5988779023439859 , -1.9728106907786414 , 0.6121863544201684 ,
	-1.7246712716663302 , -1.939199965020995 , 0.018575364152217166 ,
	1.2012724520041778 , 1.2892246845801716 , -1.727643757554169 ,
	-1.0606613028180354 , -0.7594895847911807 , 0.940448044441454 ,
	-0.5380594496163558 , -1.267145224526974 , 1.5763448526741524 ,
	0.4073834182474849 , -0.7054626718525592 , -1.9234490824213482 ,
	1.5647506616303652 , 1.2093884917144777 , -2.102359135284609 ,
	-1.7061480823447166 , -0.5244695401870394 , 0.3470674999498561 ,
	1.88819269471192 , -1.284036021000775 , -2.196739298576733 ,
	1.5101148479014979 , 0.8362643406141983 , -1.2914930059824679 ,
	0.12212200580268195 , 0.08944020420818644 , 0.03275648939425091 ,
	-3.3967549252863805 , 3.1321460307066786 , -2.6790182052952343 ,
	-0.2737460343703506 , 0.7794553737745824 , -0.39527383663395654 ,
	0.7623129340746884 , -0.019518943285166124 , 2.4060377947137885 ,
	2.3287822419247437 , -2.238774130603317 , 0.5840679716620483 ,
	2.0922798796116036 , 0.9430962738451912 , -2.166750424593194 ,
	1.4015880177978337 , 1.7637540886009968 , -2.681295764076664 ,
	-1.5086851983327452 , -1.639733399642259 , 0.676748644279972 ,
	1.4999119673752241 , 1.3523638840957775 , -2.0828720501554754 ,
	-1.7969617465018275 , -0.6237861145778117 , 0.43947272809256166 ,
	0.4829141663312104 , -0.21310204832558655 , -0.13577179420222885 ,
	-2.4596919131406425 , 2.068844957208092 , -1.6708292656945969 ,
	1.60159697335791 , 1.1781857135699831 , -1.9122737955008535 ,
	-0.7306417541756435 , -1.2443046513549227 , 1.2961770039138816 ,
	1.144834620018663 , -0.20335271454187476 , -1.167462372324861 ,
	1.6429519264614947 , -1.7386003459829027 , -1.491804846725129 ,
	0.7014001855242101 , 0.4771782721017577 , -1.1169198706730787 ,
	-2.8555745814567914 , 2.317490766500474 , -0.7162772786499436 ,
	-0.42781976406355593 , 0.7676984211895513 , -1.597732757378079 ,
	-1.0994218125263806 , -1.3285044316711025 , 0.40275483593330286 ,
	0.9322181687248499 , 0.7107739907403978 , -0.2975993924550805 ,
	-2.0010355943421527 , 1.1883922880921123 , 1.3143545834277681 ,
	-0.3931638020892527 , 0.7731655178394499 , 1.1177692467452198 ,
	0.024987447363841175 , -1.3920815840485015 , 2.095602743125138 ,
	-0.11323944770431615 , 0.6657827024658787 , 1.436966288636381 ,
	1.6277481178152962 , 0.31776238271168333 , -0.5234635363686202 ,
	1.5136076933681366 , -1.0781823618023814 , -0.6348479990508991 ,
	1.102677877108472 , 1.1626420429126714 , -1.3293883693508508 ,
	-0.9691572533687925 , -0.9439287679254668 , 0.7573390344881472 ,
	1.3765227162746776 , -1.6072947108246924 , -1.2441960144736561 ,
	-2.299625596919532 , -1.8484893770550503 , 0.0055846515011132025 ,
	-0.7888274083752244 , 1.1448178089828942 , 1.6585475262630924 ,
	-1.192412599536669 , 1.7309691936013438 , -2.684386659235871 ,
	0.4570721226576204 , 0.006203089935654438 , 1.3506480098252913 ,
	0.5432835846042493 , -0.036449230269828714 , -2.8577632635639807 ,
	0.39682878836231206 , -0.252933578352767 , -0.9702076311376171 ,
	-1.4450536372295781 , -1.250074118917691 , 0.8534220829256227 ,
	0.2785626672039654 , 0.1798723314974442 , -1.8863167243218704 ,
	1.2017994902563303 , 1.359139507332828 , -1.8605033601392025 ,
	0.5205761975962229 , 2.0387645065160496 , 2.7864062294213268 ,
	-2.1306612172416406 , 1.654349063531776 , -0.6441031891689957 ,
	1.9742154527522005 , -1.3190085109531264 , -2.2737568268941555 ,
	0.5973254019314505 , -0.48553689284427015 , -1.3151478098333236 ,
	0.9961503779492809 , 1.3806485020063959 , -1.7608232927233844 ,
	0.48934716135259504 , -0.9941537169407371 , -1.4862587991785454 ,
	0.228652426766073 , 0.12560703046878485 , -0.46256089485152496 ,
	0.8692033223220335 , 1.387106739393889 , -1.560939508452563 ,
	1.6481769031240285 , 1.2920428942470286 , -2.1504700055310333 ,
	0.35243406438339986 , 0.3301732059825849 , -2.519853242238777 ,
	-1.4531123266118662 , 0.28932064839984606 , 0.7545840516183243 ,
	-1.158753642437433 , -0.32826535786782474 , 1.043925609873369 ,
	-1.0015294887259807 , 0.63026071090551 , 0.11025672153686727 ,
	0.7635392933756445 , 0.26468295829244615 , 0.53580215899079 ,
	1.2663911727563042 , -1.6795356725207982 , -1.9585735042558852 ,
	0.7574258818971362 , 0.5059210974107693 , 0.03297909390519749 ,
	-1.1537663333392023 , 1.038219832247405 , -1.495040381284893 ,
	0.09401236101334387 , 1.1356187398529924 , -0.12067550150973338 ,
	-1.4461093312947826 , 0.18605553280456721 , 0.5895650397578898 ,
	0.5861896918128129 , -0.3558539235826712 , -2.068806112752503 ,
	0.7618697865009932 , -0.15617089693774922 , -1.7326454616527458 ,
	1.0959422325762123 , -1.5146939710870155 , 0.7808066523654318 ,
	-0.02105738231048715 , 0.38790257545972817 , -0.6937466372275372 ,
	1.0838189101998057 , 1.4557569066444795 , -2.0190511154829816 ,
	-1.004229932538435 , 1.206853641276611 , -1.3899090850608011 ,
	-0.28969470278885834 , 0.8841193377374932 , -1.6972521987751081 ,
	-1.496313770330743 , -1.7339403690463964 , 0.6897433534721699 ,
	0.08235776092526481 , -0.29806827680989123 , -0.30332973801933616 ,
	1.2969103558006352 , 1.108770189923554 , -1.5840784950253797 ,
	0.02811866843938826 , -0.3993019146343083 , -1.7812298757020484 ,
	-1.9491259243343926 , -2.1477113439803217 , 0.05941075125657891 ,
	2.5865534615566013 , -2.7206318787268158 , -1.2101872118012462 ,
	1.359828607812457 , -1.2401168279579087 , -2.034719598237685 ,
	-0.07669299583384093 , -0.1012329012456823 , 0.9181100014223551 ,
	-0.17581892471812466 , -0.27625066092064543 , 1.5650413878760174 ,
	0.6981406644608239 , -0.3186649910583945 , 0.5578895240137358 ,
	-1.3915855543305888 , -2.6247055067577447 , 0.17034618715437633 ,
	1.1590920200205337 , 0.7587254435688605 , -0.8657652500171574 ,
	1.2781026841078276 , 0.5643851736783274 , -0.5910400914810809 ,
	1.6087292685754044 , 1.2281253768982845 , -2.070974332597798 ,
	-0.16096783158257777 , 0.5440674208994474 , -1.1167926389162837 ,
	-1.210495900380837 , 1.255879200865284 , -1.4914998941236495 ,
	0.11306614486124189 , 1.0458546919621035 , -0.005536725501908646 ,
	-0.9798837719436013 , 1.1000390454207165 , -0.7878855804422348 ,
	1.263972168314998 , 1.2158450551745905 , -1.587258547235085 ,
	-1.3204523166149873 , -3.039288933201645 , 0.4242290742736061 ,
	-0.896239996028888 , -0.9892649418776902 , 1.248201155361726 ,
	-1.4538811449016582 , 0.7231551865379899 , 0.5535200449227775 ,
	-2.0328336727479916 , -1.3333656211107419 , 0.5182611633618015 ,
	1.4572741185214237 , -1.5794970074968657 , -0.3018972228948933 ,
	-0.05133053334538151 , -0.2849361913968971 , 1.9771045180204738 ,
	-0.4984260684627766 , -1.2253694136080477 , 1.4661203991581222 ,
	0.2840273595948285 , 0.13620527942684602 , 0.1019977994704182 ,
	-0.11099221766269352 , -0.9239007971210014 , 1.6519806945798154 ,
	1.4294723194013184 , -2.4399866015292497 , -0.4344691364459673 ,
	-3.168750798093503 , 2.6708931233812816 , -1.4378907384078983 ,
	0.9653171415694468 , 0.6489986665763654 , -0.3249534765141686 ,
	1.444622679234448 , -1.3007964519257498 , -0.8616838659829398 ,
	-0.5819452353618724 , 0.6196605018267485 , -0.11705817714194178 ,
	-1.0364575613093878 , -0.5807227484047671 , 1.2598212864205487 ,
	0.5853063256730974 , 0.2660301858089279 , 0.6834124362276963 ,
	-0.6911463963424044 , -1.082350766589507 , 1.2011744437228598 ,
	1.5854022712609297 , 1.5630928596330604 , -2.727498432541081 ,
	0.0876650296249637 , 0.4904757069187342 , 2.2558016496410604 ,
	-1.8333857261277322 , -2.5057249757436892 , 0.21840006188380023 ,
	0.9153534601865645 , -1.6379880188748692 , 0.029733411488719996 ,
	1.7843506163576146 , -2.7432940996483466 , -0.16150430743533262 ,
	0.8697319055390986 , 1.4017708959482482 , -1.6778699009043367 ,
	-1.4980885087358944 , -0.3910944600844284 , 0.7114655433362933 ,
	-1.3114161593143963 , -0.12755580519220655 , 1.2926055938693948 ,
	1.1058664294171923 , -0.07344705790144201 , -2.125784251714795 ,
	-0.2361293572739327 , 0.49317591473390077 , -0.02310265119300625 ,
	0.37451541195731786 , 0.7910795863191241 , 0.18814472261843518 ,
	0.1825094178501373 , 1.343660050634293 , -3.556336505201219 ,
	1.958773607105066 , 0.8657551024419513 , -2.0228367136476164 ,
	0.06521172373184136 , 1.137390983929135 , -0.11455042015689954 ,
	-1.7767947350444944 , 1.7703077317916702 , -1.5562998372102994 ,
	-1.5075541543207294 , 1.044019534785609 , 0.1549083352841112 ,
	0.8435095268913229 , -1.7184558534997794 , -0.12713959656991336 ,
	1.2739167340380861 , -3.4438777716437032 , 1.544378573181739 ,
	0.2962257783808917 , -1.0074963638632095 , 2.2587837168177827 ,
	-0.3114752381976802 , 0.5248134193177119 , 1.3385014194390505 ,
	0.9327656441211815 , -1.6584829081360517 , -0.6380283202696786 ,
	-0.6918856953292482 , -1.7573330871674566 , 1.895632670118181 ,
	0.7031705283885277 , 0.06317596206563003 , -2.6759701313662356 ,
	-0.786659067916518 , 0.008672287635122955 , 1.8553719511354512 ,
	-1.3836627356728246 , -1.8484698162503908 , 0.5323884563199665 ,
	-0.7514237778648919 , 0.40043710457207393 , 0.371343466064207 ,
	-0.15463834945743316 , 0.30466740784906343 , -0.881211064722975 ,
	1.6470440389052148 , -3.3455237031060245 , -0.32541173006558466 ,
	-0.19303861826115076 , 0.8651553775994142 , 0.564658070226773 ,
	-2.9610453744962397 , 2.8411788319597666 , -2.8672800290769 ,
	-0.9102234725901935 , 1.6681845888506581 , -3.2327920580677065 ,
	0.8704586459654822 , 0.9310384711772617 , -0.8696625023958454 ,
	0.03556838089540948 , -0.6779183960144504 , 1.5729782179484013 ,
	0.5261446699110148 , -0.48758014370899794 , 0.9342665708240578 ,
	1.2187098562462324 , -0.5985262660263531 , -0.7629250928848632 ,
	0.3589247613968396 , -0.3985804926703569 , 1.3660627434740111 ,
	0.2043513577450249 , -0.5953909830218624 , -1.438973191848022 ,
	0.633670664296464 , 0.3180943242058727 , 0.6001786286933821 ,
	0.10882707181228335 , 0.8413244987125956 , -1.0958829512293937 ,
	0.6901961511469665 , -0.828930363754834 , -0.728153295102671 ,
	-0.9262788244784119 , -1.5554843186345102 , 1.2770547032758075 ,
	0.7595865862132078 , 0.8415951233156747 , -0.5069564147995755 ,
	0.9239831523124559 , -0.11276762649295391 , -1.4867356874174988 ,
	1.4269131941907047 , -1.3194656268516836 , -0.8067472372689676 ,
	-1.13262943736167 , 1.3288223051070593 , -1.5432854849917943 ,
	1.2246681714860563 , -0.9570993904755664 , 0.012318717100882948 ,
	-1.3759914071702073 , 1.50634251983718 , -1.952298870865465 ,
	-0.3664492100658605 , 0.3929563932414524 , -0.08505208620998894 ,
	-0.5914634076025131 , 1.1435629148698605 , -2.5426739600021206 ,
	-0.9764352262045128 , -0.4686914924840597 , 1.033129091252295 ,
	1.9911286092555758 , 1.6284403425825706 , -3.277441353962943 ,
	0.587367555161759 , -0.7907801833378181 , 1.5610259740322445 ,
	0.9117100107305477 , 0.3951755751751109 , 0.08592976173859701 ,
	-1.912311073904391 , 2.126778900850857 , -2.3841970631969804 ,
	1.1065087404617513 , -1.0260685753091268 , 0.42723742590317193 ,
	1.9184637330197094 , 1.8236097626169143 , -3.385271531149489 ,
	-0.6514710482358869 , -2.3140275895355487 , 1.0708845035381698 ,
	-0.9258619539290844 , 1.0203593782356346 , -0.42292308512823307 ,
	-1.21215166167294 , -0.5090240545907487 , 0.8522178421698354 ,
	1.1752383365299144 , -1.903606929712617 , -0.4703168817576798 ,
	-3.2248796059981437 , 2.3470351602073 , -0.36790954053427705 ,
	-0.4655108430708629 , 1.4581937759541659 , -3.331964078374263 ,
	1.6976199744182026 , 0.6122096330193738 , -1.2014012742008726 ,
	-0.4424250653836217 , -0.08140699280142749 , -0.26731294320133137 ,
	-0.13841430517223807 , 0.4008358425994586 , -1.0853927267015544 ,
	0.7557192850244028 , -0.2470986518254048 , -0.8073351348784176 ,
	1.4945280678179258 , 0.6100658369242076 , -1.0484646555790418 ,
	1.3480397300088385 , 0.20993821693867254 , -0.24361692889012188 ,
	-0.2865248780772286 , 0.6740181368860345 , 0.4980804219346906 ,
	1.427201775424014 , 2.170279362437321 , -3.607699802672786 ,
	1.1866090349154845 , 0.5524103296507687 , -0.4464618421134815 ,
	0.6486543116901067 , 0.3730189933723558 , -0.8610994663119013 ,
	-1.7301802705723628 , 1.6755845161858491 , -1.5650508629251734 ,
	0.31123974894497375 , 1.1456474352956907 , -0.47867839386587085 ,
	-0.5193354272351947 , -0.4612150862928208 , 1.5608834898147919 ,
	0.9376814842126671 , -1.5802683670018984 , -0.6188853225736741 ,
	2.0683738466758443 , -1.9470342118433999 , 0.32572564002422877 ,
	1.543202571738663 , 1.3602725968418938 , -2.2511877611301463 ,
	0.45352705738024357 , 0.2893027597357303 , 0.7853673249082869 ,
	-1.8096317512529179 , 0.7554063229025134 , 0.9639697974125379 ,
	1.7715058230300655 , -3.107720456060224 , -2.154537837743421 ,
	-1.1712642239159066 , -0.3903927053167616 , 0.7684832790411574 ,
	-0.009985669446963996 , 0.8007324567833791 , 0.5315148382871384 ,
	3.1065093633197405 , -3.7530905593035944 , -1.1805308521021594 ,
	1.4453303624827294 , -1.2765955449573578 , 0.09026881122627439 ,
	0.48563261383703105 , 0.9285736016211897 , -0.36869923760417644 ,
	-1.9926168724081874 , 1.2708148225072682 , 0.28485374151793597 ,
	-0.7351289987739134 , -0.14854815146365996 , 1.7001923434610586 ,
	1.0472541290665538 , 1.0994511002344434 , -1.3182605142063017 ,
	-1.4107146008378932 , -1.0132729381573093 , 0.2366588179608331 ,
	1.11441832642768 , -0.06525736516544833 , -2.0689502728276667 ,
	1.2156159525583519 , -1.715415889551174 , -1.1515313300790522 ,
	1.4526287974767325 , -1.275659292856894 , -0.5438815532041497 ,
	-0.8508162283517582 , 1.1547673864468528 , -1.7444987855238014 ,
	-1.5183281369464927 , -2.2371584112300003 , 0.23037433846218724 ,
	-0.9986255929226018 , -0.7819789768009744 , 0.8036487373115321 ,
	1.2163567200196839 , 0.9467163978444874 , -1.2652660665602014 ,
	0.3568372444856489 , 0.6529484021760237 , -1.004144053635961 
};

#endif //BENCHMARKS_WEIGHTS_H