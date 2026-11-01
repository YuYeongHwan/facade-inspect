// 사진 데모에 쓴 공개 이미지 출처 [R33] (data/own/dirty_windows/web/SOURCES.csv에서 옮김).
// key = result.json meta.video (사진 폴더). photos_to_run은 파일명 순으로 사진 n → 외벽 F<n>.
export type PhotoSource = { file: string; author: string; license: string; page: string };

export const PHOTO_SOURCES: Record<string, PhotoSource[]> = {
  "data/own/dirty_windows/web": [
    {"file": "web_01.jpg", "author": "Sergey Ashmarin", "license": "CC BY-SA 3.0", "page": "https://commons.wikimedia.org/wiki/File:Car_with_bird_droppings_-_Rome,_Italy_-_panoramio.jpg"},
    {"file": "web_02.jpg", "author": "Atirador", "license": "CC BY 4.0", "page": "https://commons.wikimedia.org/wiki/File:Shitty_Mazda.jpg"},
    {"file": "web_03.jpg", "author": "Guillaume Vachey from Chalon sur Saone, France", "license": "CC0", "page": "https://commons.wikimedia.org/wiki/File:Renault_Twingo_(46901672205).jpg"},
    {"file": "web_04.jpg", "author": "XoMEoX", "license": "CC BY 2.0", "page": "https://commons.wikimedia.org/wiki/File:Bird%27s_Art_(14971642117).jpg"},
    {"file": "web_05.jpg", "author": "W.carter", "license": "CC BY-SA 4.0", "page": "https://commons.wikimedia.org/wiki/File:Three_gulls_on_glass_ceiling,_Kampenhof_bus_terminal,_Uddevalla.jpg"},
    {"file": "web_06.jpg", "author": "Daniel Christensen", "license": "CC BY-SA 3.0", "page": "https://commons.wikimedia.org/wiki/File:Wells_Fargo_Center_dirty_windows.jpg"},
    {"file": "web_07.jpg", "author": "Nenad Stojkovic", "license": "CC BY 2.0", "page": "https://commons.wikimedia.org/wiki/File:Dirty_back_car_window_-_49757773438.jpg"},
    {"file": "web_08.jpg", "author": "Dejan Krsmanovic", "license": "CC BY 2.0", "page": "https://commons.wikimedia.org/wiki/File:Old_Dark_and_Dirty_Window_(40343422273).jpg"},
    {"file": "web_09.jpg", "author": "Alexreavis", "license": "CC BY-SA 3.0", "page": "https://commons.wikimedia.org/wiki/File:Dusty_Window_View.JPG"},
    {"file": "web_10.jpg", "author": "Brylie", "license": "CC BY-SA 3.0", "page": "https://commons.wikimedia.org/wiki/File:Smoke_through_a_dusty_window.jpg"},
    {"file": "web_11.jpg", "author": "UncleVinny", "license": "CC BY 3.0", "page": "https://commons.wikimedia.org/wiki/File:Eiffel_Tower,_dirty_bus_window_2_-_panoramio.jpg"},
    {"file": "web_12.jpg", "author": "rachel_bernadette", "license": "CC BY-SA 2.0", "page": "https://commons.wikimedia.org/wiki/File:Though_the_dirty_window_-_Flickr_-_rachel_bernadette.jpg"},
    {"file": "web_13.jpg", "author": "Conall from Downpatrick, Northern Ireland", "license": "CC BY 2.0", "page": "https://commons.wikimedia.org/wiki/File:The_back_window_of_the_palm_house_(52712743837).jpg"},
    {"file": "web_14.jpg", "author": "Wuzur", "license": "CC BY-SA 3.0", "page": "https://commons.wikimedia.org/wiki/File:Houses_and_cars_seen_through_a_rain-streaked_car_window.jpg"},
    {"file": "web_15.jpg", "author": "Ashwini Sukhdeve", "license": "CC BY-SA 3.0", "page": "https://commons.wikimedia.org/wiki/File:The_road_to_Lavasa,_seen_through_rain-streaked_glass.jpg"},
    {"file": "web_16.jpg", "author": "W.carter", "license": "CC BY-SA 4.0", "page": "https://commons.wikimedia.org/wiki/File:East_part_of_glass_ceiling_in_Kampenhof_bus_terminal,_Uddevalla_1.jpg"},
  ],
};

/** 외벽 F<n>의 사진 출처 (출처 목록이 없는 결과면 null) */
export function photoSourceOf(video: string, facadeId: string): PhotoSource | null {
  const n = Number(facadeId.replace(/^F/, ""));
  return PHOTO_SOURCES[video]?.[n - 1] ?? null;
}
