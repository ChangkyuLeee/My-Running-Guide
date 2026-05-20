import os
import json
import requests
import numpy as np
import cv2
import tempfile
import random
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MeasureControl
from folium.plugins.draw import Draw
from pyproj import Transformer
from shapely.geometry import Point, LineString, MultiLineString
import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import warnings
# pyproj CRS 경고 숨기기
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
load_dotenv()
# load_dotenv(dotenv_path=r"C:\Users\GYU\.env")

VWORLD_KEY = os.getenv("VWORLD_KEY")
tiles = f"https://api.vworld.kr/req/wmts/1.0.0/{VWORLD_KEY}/Base/{{z}}/{{y}}/{{x}}.png" # Base, white, midnight, Hybrid
header = {'authorization': os.getenv('KAKAO_KEY')}
transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BUILTIN_GPKG_PATH = os.path.join(BASE_DIR, 'data', 'osm_walk_network.gpkg')
BUILTIN_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'Dooly.jfif')

# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite'
)
# llm = ChatOpenAI(model="gpt-5.4-mini")

st.set_page_config(page_title="그림 그려주는 러닝 가이드", layout="wide")

# # iframe 여백 제거 CSS
# st.markdown("""
# <style>
#     /* 1. iframe(지도)을 감싸는 컨테이너의 하단 여백 제거 */
#     .element-container:has(iframe) {
#         margin-bottom: 0rem !important;
#         padding-bottom: 0rem !important;
#     }
    
#     /* 2. iframe 자체를 블록 요소로 만들어 하단 미세 공백 제거 */
#     iframe {
#         display: block;
#     }
    
#     /* 3. 지도 바로 다음에 오는 표(Dataframe)와의 간격 조정 */
#     /* 필요에 따라 -1rem 값을 조절해서 간격을 더 좁히거나 넓힐 수 있습니다 */
#     .stDataFrame {
#         margin-top: -0.5rem !important; 
#     }
    
#     /* (옵션) 수직 블록 간의 기본 간격 줄이기 */
#     .stVerticalBlock > div {
#         gap: 0.5rem;
#     }
# </style>
# """, unsafe_allow_html=True)


# 데이터 캐싱
@st.cache_data(show_spinner="🗺️ 내장된 도시 도로망 데이터를 클라우드 서버 메모리에 로드하고 있습니다...")
def load_github_builtin_network(file_path):
    if not os.path.exists(file_path):
        st.error(f"🚨 [시스템 에러] GitHub 내장 데이터 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    # 💡 이미 로컬에서 u, v, length 전처리가 끝난 파일이므로
    # 클라우드 서버에서는 단순히 읽기만 하면 끝납니다. (서버 과부하 0%)
    gdf = gpd.read_file(file_path, layer='edges')
    gdf = gdf.to_crs(epsg=5179)
    return gdf

# 3. 세션 상태에 데이터 상주
if "walk_network" not in st.session_state:
    st.session_state["walk_network"] = load_github_builtin_network(BUILTIN_GPKG_PATH)

# 4. 전역 변수 매핑
walk_network = st.session_state["walk_network"]

#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--##--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#

# 헬퍼 함수들
def transform_to_latlon(meter_coords, center_lat, center_lon):
    # 1. 좌표계 정의 
    # EPSG:4326(경위도) -> EPSG:5179(도로명주소/네이버/카카오 등에서 쓰는 미터 단위 좌표계)
    # 한국 지역이 아니라면 EPSG:3857(구글맵 전세계 표준) 등을 사용해도 됩니다.
    transformer_to_m = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    transformer_to_deg = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

    # 2. 중심점(위경도)을 미터 단위 좌표로 변환
    center_x, center_y = transformer_to_m.transform(center_lon, center_lat)

    final_latlon_coords = []

    # 3. 각 미터 좌표에 중심점 미터 좌표를 더한 뒤 다시 경위도로 역변환
    for pt in meter_coords:
        real_x = center_x + pt['x']
        real_y = center_y + pt['y']
        
        # 미터 -> 경위도 변환
        lon, lat = transformer_to_deg.transform(real_x, real_y)
        final_latlon_coords.append({'lat': lat, 'lon': lon})

    return final_latlon_coords


def get_shape_coordinates(perimeter, shape):
    """
    둘레 길이와 모양을 입력받아 중심 (0,0) 기준의 좌표 리스트를 반환하는 함수
    """
    system_instructions = (
        "당신은 기하학 및 좌표 계산 전문가입니다. 사용자가 입력한 둘레 길이와 모양에 맞춰 "
        "중심 좌표 (0, 0)를 기준으로 도형의 윤곽선을 그리기 위한 x, y 좌표 리스트를 계산합니다.\n"
        f"중심 좌표: (0, 0)\n"
        f"모양: {shape}\n"
        f"둘레 길이: {perimeter}m\n"
        "[매우 중요한 지침]\n"
        "1. 곡선이나 복잡한 모양의 디테일을 표현하기 위해 최소 50개 이상의 좌표(점)를 생성하세요.\n"
        "2. 테두리 선은 반드시 한 바퀴(1회전)만 돌아야 합니다. 겹치는 선이나 두 번 도는 현상이 절대 없도록 하세요.\n"
        "3. 윤곽선이 궤도를 한 번만 완성하면 즉시 좌표 생성을 멈추세요.\n"
        "4. 입력된 둘레 길이(m)에 맞게 각 좌표의 스케일(비율)을 정확히 곱해서 크기를 키우세요.\n"
        "5. 닫힌 다각형이 되도록 마지막 좌표는 반드시 첫 번째 좌표와 동일해야 합니다.\n\n"
        "- 출력은 JSON object 형식이어야 하며, 단일 키 'coordinates'를 가집니다.\n"
        "- 'coordinates'의 값은 [{'x': 10.5, 'y': 20.0}, ...] 형태의 리스트입니다.\n"
        "- 반드시 순수 JSON 문자열만 출력하세요. 마크다운이나 다른 설명은 절대 넣지 마세요."
    )
    
    # LLM에 프롬프트 전달 및 응답 받기
    gpt_response = llm.invoke(system_instructions)

    # JSON 파싱
    gpt_result = json.loads(gpt_response.content) # GPT 응답을 JSON으로 변환    
    coordinates = gpt_result.get('coordinates', [])
        
    return coordinates


def geocode_keyword(header, destination):
    loc_info = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json?&query=' + destination, # 관광지 검색
                                headers=header).json()
    loc = loc_info['documents'][0]

    # 카카오 결과
    place_name = loc['place_name']
    address_name = loc['address_name']
    coord_x = loc['x']
    coord_y = loc['y']

    return place_name, address_name, coord_x, coord_y


def resample_line(line, interval=10):
    """
    LineString을 입력받아 지정된 간격(interval, m)으로 버텍스를 재생성합니다.
    """
    # 1. 선의 전체 길이 계산
    total_length = line.length
    
    # 2. 전체 길이를 간격으로 나누어 지점들 생성
    distances = np.arange(0, total_length, interval)
    
    # 3. 각 지점(distance)에 해당하는 좌표를 추출하여 새로운 점 생성
    new_points = [line.interpolate(distance) for distance in distances]
    
    # 마지막 끝점 추가 (누락 방지)
    new_points.append(Point(line.coords[-1]))
    
    return LineString(new_points)


def get_random_offset(radius_m):
    """반경 radius_m 내의 무작위 X, Y 오프셋 생성"""
    r = radius_m * np.sqrt(random.random())
    theta = random.random() * 2 * np.pi
    return r * np.cos(theta), r * np.sin(theta)


def node_skipping_smoothing(G, path, max_skip=1000, efficiency_ratio=0.95):
    if len(path) < 3:
        return path

    smoothed_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        curr_node = path[i]
        skipped = False
        
        # 너무 멀리 점프하면 경로 의도가 훼손되므로 max_skip 범위 내에서 찾음
        look_ahead = min(i + max_skip, len(path) - 1)
        
        for j in range(look_ahead, i + 1, -1):
            next_node = path[j]
            # 1. 두 노드 사이에 직접적인 도로(Edge)가 존재하는지 확인
            if G.has_edge(curr_node, next_node):
                # 2. 지름길(직선 도로)의 길이 계산
                direct_dist = G[curr_node][next_node].get('weight', 0)
                
                # 3. 원래 가려던 경로(i부터 j까지)의 누적 길이 계산
                original_dist = 0
                for k in range(i, j):
                    original_dist += G[path[k]][path[k+1]].get('weight', 0)
                
                # 4. 조건 확인: 지름길이 유의미하게 짧은가?
                # P턴 구간이라면 original_dist가 훨씬 길기 때문에 조건이 참이 됨
                if direct_dist < (original_dist * efficiency_ratio):
                    smoothed_path.append(next_node)
                    i = j
                    skipped = True
                    break
        
        if not skipped:
            i += 1
            smoothed_path.append(path[i])
    return smoothed_path


def find_running_route(draw_gdf_5179, walk_network=walk_network):
    bounds = draw_gdf_5179.total_bounds

    # 2. 상자에 여유(Padding) 추가
    padding = 500
    minx = bounds[0] - padding
    miny = bounds[1] - padding
    maxx = bounds[2] + padding
    maxy = bounds[3] + padding

    walk_network_5179 = walk_network.cx[minx:maxx, miny:maxy]

    # 선 주변 일정 반경을 면(Polygon)으로 확장합니다.
    tolerance_meters = 200
    draw_buffer = draw_gdf_5179.copy()
    draw_buffer['geometry'] = draw_gdf_5179.geometry.buffer(tolerance_meters)


    # 2. 공간 결합(Spatial Join)을 통해 버퍼 내부에 포함되거나 걸치는 도로 추출
    extracted_network = gpd.sjoin(walk_network_5179, draw_buffer, how='inner', predicate='intersects')

    # 중복된 인덱스 제거 (결합 과정에서 발생할 수 있음)
    extracted_network = extracted_network.loc[~extracted_network.index.duplicated(keep='first')]


    # 1. 사용자가 그린 선의 실제 좌표 추출
    coords = list(draw_gdf_5179.geometry.iloc[0].coords)
    user_start_pt = Point(coords[0])
    user_end_pt = Point(coords[-1])

    user_start_P = gpd.GeoDataFrame(geometry=[user_start_pt], crs=5179).to_crs(4326)
    user_end_P = gpd.GeoDataFrame(geometry=[user_end_pt], crs=5179).to_crs(4326)


    # 2. 모든 도로(edges)의 시작점과 끝점을 모아 '노드 집합' 만들기
    # 각 edge의 첫 좌표와 끝 좌표를 추출
    edge_points = []
    for geom in walk_network_5179.geometry:
        if geom.geom_type == 'LineString':
            edge_points.append(Point(geom.coords[0]))
            edge_points.append(Point(geom.coords[-1]))


    # # 중복 제거를 위해 다시 GDF로 변환 (속도를 위해 MultiPoint 활용 가능)
    # nodes_from_edges = gpd.GeoDataFrame(geometry=edge_points, crs=walk_network_5179.crs).drop_duplicates()


    # # 3. 가장 가까운 노드 ID(또는 좌표) 찾기
    # start_node_geom = nodes_from_edges.geometry.loc[nodes_from_edges.distance(user_start_pt).idxmin()]
    # end_node_geom = nodes_from_edges.geometry.loc[nodes_from_edges.distance(user_end_pt).idxmin()]


    ## 가까운 도로 추출 방식
    # 1. 추출된 도로들의 중심점 계산
    extracted_edges = extracted_network.copy()
    extracted_edges['centroid'] = extracted_edges.geometry.centroid

    # 2. 사용자가 그린 선(LineString) 준비
    user_line = draw_gdf_5179.geometry.iloc[0]

    # 각 도로(Edge)의 중심점과 사용자 선 사이의 최단 거리 계산
    # .distance() 함수는 두 객체 사이의 최단 거리를 반환함
    extracted_edges['dist_to_draw'] = extracted_network.geometry.centroid.apply(lambda x: user_line.distance(x))

    max_dist = 100
    # 설정한 임계값 이내의 도로만 남김
    extracted_edges = extracted_edges[extracted_edges['dist_to_draw'] <= max_dist]


    # 3. 각 중심점이 사용자의 선 상에서 어느 위치(투영 거리)에 있는지 계산
    # user_line.project(point)는 선의 시작점으로부터 해당 점까지의 거리를 반환함
    extracted_edges['project_dist'] = extracted_edges['centroid'].apply(lambda x: user_line.project(x))

    # 4. 투영 거리 순으로 정렬 (사용자가 그린 방향대로 정렬됨)
    ordered_edges = extracted_edges.sort_values(by='project_dist')


    waypoints = ordered_edges['centroid'].tolist()


    # 1. MultiGraph 생성 (OSM 데이터는 양방향이나 중복 선이 있을 수 있음)
    G = nx.Graph()

    # 추출된 도로의 인덱스 세트 (빠른 비교를 위해)
    extracted_inds = set(extracted_network.index)

    for idx, row in walk_network_5179.iterrows():
        # 기본 가중치는 실제 도로의 길이(length)
        weight = row.geometry.length
        
        # 만약 이 도로가 사용자가 그린 영역(extracted)에 있다면 가중치를 대폭 낮춤
        # 가중치가 낮을수록 알고리즘은 이 길을 '가까운 길'로 인식하여 우선 선택함
        if idx in extracted_inds:
            weight = weight * 0.1  # 10배 더 매력적인 길로 설정
        else:
            weight = weight * 1  # 추출되지 않은 길은 가급적 피하도록 설정

        G.add_edge(row['u'], row['v'], weight=weight, original_index=idx)    

        # u_node = row['u'] if 'u' in row else row.get('index_left', idx)
        # v_node = row['v'] if 'v' in row else row.get('index_right', idx)
        # G.add_edge(u_node, v_node, weight=weight, original_index=idx)


    G_weighted = G.copy()


    # 모든 edge의 u, v 노드 좌표를 기반으로 탐색 (이미 nodes GDF가 있다면 그것을 사용)
    # 여기서는 edges 데이터만 있으므로 각 edge의 첫 지점을 노드 대표값으로 사용
    node_points = walk_network_5179.copy()
    node_points['geometry'] = node_points.geometry.apply(lambda x: Point(x.coords[0]))


    node_ids = []
    for pt in waypoints:
        # 각 경유지(중심점)에서 가장 가까운 도로의 시작 노드(u) ID 추출
        nearest_idx = node_points.distance(pt).idxmin()
        node_ids.append(walk_network_5179.loc[nearest_idx, 'u'])

    # 연속된 중복 노드 제거 (예: [1, 2, 2, 3] -> [1, 2, 3])
    cleaned_node_ids = []
    if node_ids:
        cleaned_node_ids.append(node_ids[0])
        for i in range(1, len(node_ids)):
            if node_ids[i] != node_ids[i-1]:
                cleaned_node_ids.append(node_ids[i])

    ## 경로 찾기
    final_path_nodes = []
    for i in range(len(cleaned_node_ids) - 1):
        origin = cleaned_node_ids[i]
        dest = cleaned_node_ids[i+1]
        
        try:
            # 커스텀 가중치를 기준으로 최단 경로 계산
            sub_path = nx.shortest_path(G_weighted, source=origin, target=dest, weight='weight')
            
            # 경로가 겹치지 않게 추가 (다음 구간의 시작점이 현재 구간의 끝점과 같으므로)
            if i == 0:
                final_path_nodes.extend(sub_path)
            else:
                final_path_nodes.extend(sub_path[1:])
        except nx.NetworkXNoPath:
            # 만약 끊긴 구간이 있다면 건너뜀
            continue

    # 완성된 전체 경로에서 불필요한 꺾임이나 뺑뺑이를 네트워크 기반으로 제거
    if len(final_path_nodes) > 2:
        final_path_nodes = node_skipping_smoothing(G_weighted, final_path_nodes)

    # 에지 지오메트리 생성
    edge_geometries = []

    for i in range(len(final_path_nodes) - 1):
        u, v = final_path_nodes[i], final_path_nodes[i+1]
        
        # 그래프에서 u, v 사이의 edge 정보를 가져옴
        # MultiGraph일 수 있으므로 가장 짧은 거리나 첫 번째 데이터를 선택
        edge_data = G_weighted.get_edge_data(u, v)
        
        if edge_data:
            # 원본 edges 데이터에서 해당 edge의 geometry를 찾아 추가
            # u, v 순서가 바뀌었을 수도 있으므로 양방향 체크
            mask = ((walk_network_5179['u'] == u) & (walk_network_5179['v'] == v)) | \
                    ((walk_network_5179['u'] == v) & (walk_network_5179['v'] == u))
            
            if any(mask):
                geom = walk_network_5179.loc[mask, 'geometry'].iloc[0]
                edge_geometries.append(geom)

    # 들어갔던 길을 지움 (현재 길도 추가 안 함)
    stack = []
    for edge in edge_geometries:
        if stack and edge.equals(stack[-1]):
            stack.pop() 
        else:
            stack.append(edge)

    # 모든 선분을 하나로 합침
    combined_line = MultiLineString(stack)
    route_gdf_5179 = gpd.GeoDataFrame({'geometry': [combined_line]}, crs=5179)
    route_gdf_5179['Total_Length'] = route_gdf_5179.geometry.length

    # 시각화를 위해 다시 위경도(4326)로 변환
    # route_gdf_4326 = route_gdf_5179.to_crs(epsg=4326)
    
    return route_gdf_5179, user_start_P, user_end_P


def directed_mhd(line_a, line_b, num_samples=50):
    """line_a의 샘플 점들로부터 line_b까지의 최단 거리들의 평균을 구합니다."""
    # 1. 두 선의 길이에 따라 일정 간격으로 점을 샘플링합니다.
    distances_a = np.linspace(0, line_a.length, num_samples)
    points_a = [line_a.interpolate(d) for d in distances_a]
    
    # 2. 각 점과 line_b 사이의 최단 거리를 구한 후 평균을 냅니다.
    min_distances = [p.distance(line_b) for p in points_a]
    return np.mean(min_distances)

def modified_hausdorff_distance(line_1, line_2, num_samples=50):
    """두 LineString 사이의 Modified Hausdorff Distance(MHD)를 계산합니다."""
    # 대칭성을 만족하기 위해 양방향(1->2, 2->1) 중 최댓값을 취합니다.
    mhd_12 = directed_mhd(line_1, line_2, num_samples)
    mhd_21 = directed_mhd(line_2, line_1, num_samples)
    return max(mhd_12, mhd_21)


def evaluate_routes(draw_gdf_5179, iterations=30, walk_network=None):
    results = []
    geometry = []
    
    # original_shape_5179: LLM이 준 좌표 리스트
    target_line_00 = draw_gdf_5179.geometry[0]

    for i in range(iterations):
        # 1. 1km 반경 내 무작위 이동 오프셋 결정
        off_x, off_y = get_random_offset(1000)
        
        # 3. 모양 이동
        shifted_coords = [(p[0] + off_x, p[1] + off_y) for p in target_line_00.coords]
        target_line_shifted = LineString(shifted_coords)
        target_line_shifted_gdf = gpd.GeoDataFrame(geometry=[target_line_shifted], crs=5179)
        
        # 4. [이 부분에 기존의 도로 매칭 로직 수행]
        matched_route_5179, user_start_P, user_end_P = find_running_route(target_line_shifted_gdf, walk_network)
        
        # 5. ⭐ Modified Hausdorff 거리 계산으로 교체 (기존 내장함수 대신 커스텀 함수 사용)
        # num_samples 값을 늘릴수록 정밀해지지만 연산 속도를 위해 50~100 사이를 추천합니다.
        mhd_dist = modified_hausdorff_distance(target_line_shifted, matched_route_5179.geometry[0], num_samples=50)
        
        results.append({
            'id': i + 1,
            'offset': (off_x, off_y),
            'mhd_dist': mhd_dist,              # 변수명 변경 (mhd_dist)
            'user_start_P': user_start_P,
            'user_end_P': user_end_P,
            'Total_Length': matched_route_5179.geometry.length[0]
        })
        geometry.append(matched_route_5179.geometry[0])

    # 6. 거리 기준 오름차순 정렬 (MHD 거리가 짧을수록 1등)
    ranking_gdf = gpd.GeoDataFrame(data=results, geometry=geometry, crs=5179)
    ranking_gdf = ranking_gdf.sort_values(by='mhd_dist').reset_index(drop=True) # 정렬 기준 변경
    ranking_gdf['rank'] = ranking_gdf.index + 1
    
    return ranking_gdf


def get_color(rank_index):
    """순위에 맞는 HEX 색상을 반환 (0부터 시작하는 index)"""
    rgba = cmap(rank_index)
    return mcolors.rgb2hex(rgba)


def extract_path_from_image(image_path, coord_x, coord_y, scale_meters):
    center_x, center_y = transformer_to_m.transform(coord_x, coord_y)

    # 1. 이미지 읽기 및 그레이스케일 변환
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 읽을 수 없습니다: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. 이미지 이진화 (배경과 객체 분리)
    # 이미지 특성에 따라 THRESH_BINARY_INV 등을 조절해야 할 수 있습니다.
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 3) Canny Edge Detection
    # 3) Otsu 알고리즘으로 이미지 최적의 Threshold 임계값(high_thresh) 자동 계산
    # 이 방식은 이미지의 히스토그램을 분석해 배경과 객체를 나누는 최적의 기준점을 스스로 찾습니다.
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * 0.5  # 보통 높은 임계값의 절반을 낮은 임계값으로 설정

    # 4) 계산된 자동 임계값으로 Canny Edge 실행 (절대 에러가 나지 않는 구조)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # 5) 선 연결을 위한 모폴로지 연산 (커널 크기를 줄여 선이 지워지는 걸 방지)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 닫기(Closing) 연산만 적용하여 끊어진 선들을 부드럽게 이어줍니다.
    processed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. 윤곽선(Contour) 추출
    # 에러 방지를 위해 가장 바깥쪽 외곽선(RETR_EXTERNAL)과 더불어 전체 윤곽선(RETR_LIST) 후보군을 유연하게 잡습니다.
    contours, _ = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 만약 외곽선이 안 잡혔다면 전체 선(RETR_LIST) 중에서 다시 찾습니다.
    if not contours:
        contours, _ = cv2.findContours(processed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
    if not contours:
        # 이 단계에서도 안 잡힌다면 진짜 이미지에 아무 선이 없는 상태입니다. (전체 흰색 또는 전체 검은색 화면)
        raise ValueError("이미지 분석 실패: 이미지에서 아무런 시각적 경계선도 찾을 수 없습니다. 다른 이미지를 사용해보세요.")

    # 가장 큰 객체의 윤곽선을 선택
    longest_contour = max(contours, key=len)
    # 픽셀 좌표 추출 (N, 2) 형태
    pixel_coords = longest_contour.reshape(-1, 2)
    # 폐곡선을 만들기 위해 출발점을 마지막에 추가
    pixel_coords = np.vstack([pixel_coords, pixel_coords[0]])

    # 4. 지리 좌표계(미터)로 변환을 위한 스케일링 및 정규화
    # 픽셀 좌표의 중심을 (0,0)으로 맞춥니다.
    min_p = pixel_coords.min(axis=0)
    max_p = pixel_coords.max(axis=0)
    img_w = max_p[0] - min_p[0]
    img_h = max_p[1] - min_p[1]

    # 안전장치: 점 하나짜리 이미지일 경우 분모가 0이 되는 것 방지
    if img_w == 0: img_w = 1
    if img_h == 0: img_h = 1

    normalized_coords = (pixel_coords - min_p) / [img_w, img_h] # 0~1 사이로 정규화
    normalized_coords -= 0.5 # 중심을 0으로 이동 (-0.5 ~ 0.5)

    # OpenCV는 Y축이 아래로 갈수록 증가하므로, 지도 좌표계와 맞추기 위해 Y축 뒤집기
    normalized_coords[:, 1] = -normalized_coords[:, 1]

    # 5. 변환된 미터 중심점(center_x, center_y)과 scale_meters 적용
    aspect_ratio = img_w / img_h
    if aspect_ratio > 1:
        scale_x = scale_meters
        scale_y = scale_meters / aspect_ratio
    else:
        scale_x = scale_meters * aspect_ratio
        scale_y = scale_meters
        
    geo_coords = []
    for x_norm, y_norm in normalized_coords:
        real_x = center_x + (x_norm * scale_x)
        real_y = center_y + (y_norm * scale_y)
        geo_coords.append((real_x, real_y))
        
    # 6. Shapely LineString 및 GeoDataFrame 생성 (5179 좌표계 설정)
    target_line = LineString(geo_coords)

    # 과도하게 많은 마디 점들을 정리해서 연산 효율 향상 (5m 단위 단순화)
    if len(geo_coords) > 100:
        target_line = target_line.simplify(tolerance=5.0, preserve_topology=True)

    image_gdf_5179 = gpd.GeoDataFrame(geometry=[target_line], crs=5179)

    return image_gdf_5179


#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--##--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#

# --- 1. 페이지 설정 및 초기화 ---
st.title("🏃‍♂️ 나만의 GPS 아트 러닝 코스 메이커")

# 세션 상태(Session State) 설계
if "step" not in st.session_state:
    st.session_state["step"] = 1  # 1: 입력 단계, 2: 코스 생성 단계
if "route_mode" not in st.session_state:
    st.session_state["route_mode"] = None
if "generated_route" not in st.session_state:
    st.session_state["generated_route"] = None


# 현재 세션 모드 임시 변수에 대입 (안전하게 미리 선언)
current_mode = st.session_state["route_mode"]

# --- 2. 사이드바: 기본 러닝 정보 입력 (Step 1) ---
with st.sidebar:
    st.header("📋 러닝 코스 조건 설정")
    
    # 위치 입력 (기본값 설정)
    destination = st.text_input("📍 달리고 싶은 지역/위치 입력", value="언주역")
    
    # 거리 선택 (슬라이더)
    target_distance = st.slider("🛣️ 목표 러닝 거리 (km)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    
    st.markdown("---")
    
    # 코스 생성 방식 선택 인터페이스
    st.subheader("🎨 코스 생성 방식 선택")
    mode = st.radio(
        "원하는 방식을 선택하세요:",
        (
            "🤖 AI에게 키워드로 코스 추천받기",
            "🖼️ 이미지 실루엣으로 코스 따기",
            "✏️ 지도 위에 직접 가이드라인 그리기"
        )
    )
    
    # 모드 선택 매핑
    if "AI" in mode:
        st.session_state["route_mode"] = "AI"
    elif "이미지" in mode:
        st.session_state["route_mode"] = "IMAGE"
    elif "직접" in mode:
        st.session_state["route_mode"] = "DRAW"

    # ⭐ 변수가 실시간 라디오버튼 조작을 반영하도록 동기화
    current_mode = st.session_state["route_mode"]

    st.markdown("---")
    # 조건 확정 버튼
    if st.button("🚀 코스 빌더 가동하기", use_container_width=True):
        st.session_state["step"] = 2
        st.success(f"{destination} 주변 {target_distance}km 코스 생성을 시작합니다!")

if current_mode == "AI":
    st.markdown("### 🤖 AI 맞춤형 코스 가이드 생성")
    st.write("러닝 코스의 모양을 입력하면 AI가 기하학적 좌표를 계산해 매칭 도로를 찾습니다.")
    
    # 1. 입력 폼 컴포넌트
    target_shape = st.text_input("🎨 원하는 러닝 코스 모양 입력", value="하트")
    # [연동] 사용자가 설정한 목표 거리를 미터 단위로 환산 (예: 5.0km -> 5000m)
    target_perimeter_m = target_distance * 1000
    
    if st.button("AI 코스 생성하기", type="primary"):
        # 안전한 에러 처리를 위한 try-except 블록
        try:
            # [단계 1] 카카오 API를 통한 장소 검색 (지오코딩)
            with st.spinner(f"📍 입력하신 장소('{destination}') 위치 탐색 중..."):
                # geocode_keyword 함수 호출
                place_name, address_name, coord_x, coord_y = geocode_keyword(
                    header=header, 
                    destination=destination
                )
                st.success(f"🔍 장소 확인: {place_name} ({address_name})")

            # [단계 2] LLM 가동하여 (0,0) 기준 가이드 좌표 획득
            with st.spinner(f"🤖 AI가 모양('{target_shape}') 분석 및 기하학적 좌표 계산 중..."):
                coords_gpt = get_shape_coordinates(target_perimeter_m, target_shape)

            # [단계 3] AI 좌표를 검색된 장소의 미터 단위 공간(5179)으로 이동 및 경위도 변환
            with st.spinner("🌍 추출된 좌표를 선택하신 장소 위로 이동하는 중..."):
                # 제공해주신 transform_to_latlon 함수 호출 (결과: {'lat': ..., 'lon': ...} 리스트)
                translated_coords = transform_to_latlon(coords_gpt, coord_y, coord_x)
                
                geometry = []
                for point in translated_coords:
                    geometry.append(Point(point['lon'], point['lat']))

                one = gpd.GeoDataFrame(geometry=[LineString(geometry)], crs=4326)
                draw_gdf_5179 = one.to_crs(epsg=5179)
                # 적용 예시: 사용자가 그린 선을 10m 간격으로 촘촘하게 재구성
                draw_gdf_5179['geometry'] = draw_gdf_5179.geometry.apply(lambda x: resample_line(x, interval=10))

            # [단계 4] 도로망 네트워크 매칭 엔진 가동
            with st.spinner("🛣️ 실제 도로 매칭 중..."):
                num_ranks = 5
                ranking_results = evaluate_routes(draw_gdf_5179, iterations=num_ranks, walk_network=walk_network)
                ranking_results_4326 = ranking_results.to_crs(4326)

                matched_route_5179, user_start_P, user_end_P = find_running_route(draw_gdf_5179, walk_network=walk_network)
                matched_route_4326 = matched_route_5179.to_crs(4326)
                
                # 지도시각화와 통계 산출을 위해 세션 상태에 저장
                st.session_state["one"] = one
                st.session_state["draw_gdf_5179"] = draw_gdf_5179
                st.session_state["ranking_results_4326"] = ranking_results_4326
                st.session_state["matched_route_5179"] = matched_route_5179
                st.session_state["matched_route_4326"] = matched_route_4326
                st.session_state["num_ranks"] = num_ranks

                st.session_state["generated_route"] = True
                
                st.balloons() # 매칭 성공 효과
                
        except IndexError:
            st.error("❌ 카카오 장소 검색 결과가 없습니다. 보다 정확한 명칭(예: 역이름, 건물명)을 입력해 주세요.")
        except Exception as e:
            st.error(f"❌ 코스 생성 중 오류가 발생했습니다: {e}")

elif current_mode == "IMAGE":
    st.markdown("### 🖼️ 이미지 실루엣 코스 변환기")
    st.write("원하는 캐릭터 실루엣, 도형 아이콘(PNG/JPG)을 업로드하면 경계선을 자동으로 인식해 실제 도로망에 붙여줍니다.")

    # 1. 파일 업로더 컴포넌트 배치
    uploaded_file = st.file_uploader("📂 경계를 추출할 실루엣 이미지를 올려주세요.", type=["png", "jpg", "jpeg", "jfif", "bmp", "tif"])
    
    # 💡 기본 이미지 세팅 지점
    default_image_path = "data/Dooly.jfif"  # 프로젝트 폴더 내에 함께 업로드할 기본 파일명
    target_image_found = False

    if uploaded_file is not None:
        st.success(f"✔️ 이미지 파일 로드 완료: {uploaded_file.name}")        
        target_image_found = True
    elif os.path.exists(default_image_path):
        st.info("💡 업로드된 파일이 없어 시스템 기본 이미지(둘리)를 사용합니다.")
        target_image_found = True

    if target_image_found:
        target_perimeter_m = target_distance * 1000
        scale_meters = target_perimeter_m/3.8

        # 2. 코스 빌드 실행 버튼
        if st.button("🖼️ 이미지 모양대로 코스 매칭하기", type="primary"):
            temp_img_path = None
            try:
                # 💡 [핵심 안전장치] 업로드된 파일 스트림을 임시 디스크 경로로 변환
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_img_path = tmp_file.name
                else:
                    temp_img_path = default_image_path  # 기본 사진 경로 바인딩
                    
                # [단계 1] 카카오 API를 통해 중심점을 잡을 사용자 장소 변환
                with st.spinner(f"📍 코스를 얹을 중심 위치('{destination}') 탐색 중..."):
                    place_name, address_name, coord_x, coord_y = geocode_keyword(
                        header=header, 
                        destination=destination
                    )
                
                # [단계 2] 이미지 외곽선 알고리즘 호출하여 5179 가이드라인 생성
                with st.spinner("🔍 이미지 분석 및 기하학적 윤곽선(Contour) 정밀 추출 중..."):
                    draw_gdf_5179 = extract_path_from_image(
                        image_path=temp_img_path,
                        coord_x=float(coord_x),
                        coord_y=float(coord_y),
                        scale_meters=scale_meters
                    )
                    
                    # 하단 지도 표출을 위해 필요한 원본 위경도 데이터 변환 및 세션 박스 동기화
                    one = draw_gdf_5179.to_crs(epsg=4326)
                    draw_gdf_5179['geometry'] = draw_gdf_5179.geometry.apply(lambda x: resample_line(x, interval=10))

                # [단계 3] 내장된 도로 네트워크 매칭 엔진 가동 (이전 AI 모드 하단에 구현해둔 로직과 구조 싱크 정렬)
                with st.spinner("🛣️ 가이드라인 주변의 최적 직선 도로 탐색 및 가중치 조율 중..."):
                    num_ranks = 5
                    ranking_results = evaluate_routes(draw_gdf_5179, iterations=num_ranks, walk_network=walk_network)
                    ranking_results_4326 = ranking_results.to_crs(4326)

                    matched_route_5179, user_start_P, user_end_P = find_running_route(draw_gdf_5179, walk_network=walk_network)
                    matched_route_4326 = matched_route_5179.to_crs(4326)
                    
                    # 지도시각화와 통계 산출을 위해 세션 상태에 저장
                    st.session_state["one"] = one
                    st.session_state["draw_gdf_5179"] = draw_gdf_5179
                    st.session_state["ranking_results_4326"] = ranking_results_4326
                    st.session_state["matched_route_5179"] = matched_route_5179
                    st.session_state["matched_route_4326"] = matched_route_4326
                    st.session_state["num_ranks"] = num_ranks

                    st.session_state["generated_route"] = True

                    st.balloons() # 매칭 성공 효과
                    
            except Exception as e:
                # 에러 추적 로그 활성화
                import traceback
                st.error("❌ 이미지 기반 코스 생성 중 오류 발생")
                st.code(traceback.format_exc(), language="python")
            finally:
                # 사용이 끝난 디스크의 임시 파일 안전하게 삭제하여 흔적 제거
                if temp_img_path and os.path.exists(temp_img_path):
                    if 'Dooly' not in temp_img_path:
                        os.remove(temp_img_path)

elif current_mode == "DRAW":
    st.markdown("### 🎨 지도에 직접 코스 그리기")
    st.write("우측 상단의 선 그리기 도구(Polyline)를 선택해 지도 위에 원하는 러닝 코스를 직접 그려보세요.")

    # st_folium 하단과 주변의 불필요한 마진/패딩을 강제로 제거하는 CSS 주입
    st.html("""
        <style>
            /* st_folium이 감싸고 있는 요소의 하단 공백 강제 제거 */
            .stFolium {
                margin-bottom: -30px !important;
            }
            /* element 간의 세로 간격을 좁힘 */
            div[data-testid="stVerticalBlock"] > div {
                padding-bottom: 0.5rem !important;
            }
        </style>
    """)

    # 1. 중심점 좌표 가져오기 (카카오 API 호출 결과 혹은 기본 세션 활용)
    # 기존 코드의 지오코딩 딜레이를 방지하기 위해 사용자가 입력한 destination 검색 결과를 활용합니다.
    try:
        place_name, address_name, coord_x, coord_y = geocode_keyword(header=header, destination=destination)
        center_coords = [float(coord_y), float(coord_x)]
    except:
        # 카카오 API 호출 실패 시 기본 서울 중심점 가이드
        center_coords = [37.5665, 126.9780]

# 2. Folium 도화지 맵 생성
    m_draw = folium.Map(location=center_coords, zoom_start=15)
    
    folium.TileLayer(
        tiles=tiles,
        overlay=True,
        control=False,
        attr="VWorld",
        name="Vworld"
    ).add_to(m_draw)

    # 3. 그리기(Draw) 플러그인 장착 (export 옵션은 웹 스트림이므로 False 처리)
    Draw(
        export=False,
        position="topright",
        draw_options={
            "polygon": False,
            "polyline": True,      # 선 그리기만 활성화
            "rectangle": False,
            "marker": False,
            "circle": False,
            "circlemarker": False,
        },
        edit_options={"poly": {"allowIntersection": False}},
    ).add_to(m_draw)

    # 4. 💡 핵심: Streamlit 화면에 지도를 그리고 사용자의 마우스 움직임(데이터)을 실시간 리턴받음
    # key를 지정하여 지도가 리렌더링될 때 상태를 보존합니다.
    with st.container():
        output = st_folium(m_draw, use_container_width=True, height=500, key="user_draw_map")
    st.divider() # 구분선

    # 5. 사용자가 지도의 그리기 도구로 선을 완성했는지 감지하는 컴포넌트 엔진
    drawn_geom = None
    if output and "all_drawings" in output and output["all_drawings"]:
        # 가장 최근에 그린 그리기 객체(Feature) 추출
        last_drawing = output["all_drawings"][-1]
        
        if last_drawing and "geometry" in last_drawing:
            geom_type = last_drawing["geometry"]["type"]
            coords = last_drawing["geometry"]["coordinates"]
            
            if geom_type == "LineString":
                # 사용자가 그린 위경도 기반 Shapely LineString 변환
                drawn_geom = LineString(coords)
                st.info("✔️ 코스 선이 감지되었습니다! 아래 버튼을 누르면 실제 도로와 매칭을 시작합니다.")
            else:
                st.warning(f"⚠️ 선(Polyline) 도구만 지원합니다. 현재 입력 타입: {geom_type}")

    # 6. 매칭 실행 파이프라인 트리거 버튼
    if drawn_geom:
        if st.button("🛣️ 내가 그린 코스대로 도로 매칭하기", type="primary", use_container_width=True):
            try:
                # [단계 1] 그리기 데이터를 기반으로 GeoDataFrame 빌드
                one = gpd.GeoDataFrame(geometry=[drawn_geom], crs=4326)
                draw_gdf_5179 = one.to_crs(epsg=5179)
                
                # 기존 인터벌 리샘플링 적용
                draw_gdf_5179['geometry'] = draw_gdf_5179.geometry.apply(lambda x: resample_line(x, interval=10))
                
                # [단계 2] 매칭 엔진 가동 스피너 구역 (AI/IMAGE 모드와 동일한 구조 바인딩)
                with st.spinner("🛣️ 그린 선을 기반으로 실제 러닝 도로망 데이터를 분석 중..."):
                    num_ranks = 5
                    ranking_results = evaluate_routes(draw_gdf_5179, iterations=num_ranks, walk_network=walk_network)
                    ranking_results_4326 = ranking_results.to_crs(4326)

                    matched_route_5179, user_start_P, user_end_P = find_running_route(draw_gdf_5179, walk_network=walk_network)
                    matched_route_4326 = matched_route_5179.to_crs(4326)
                    
                    # 지도시각화와 통계 산출을 위해 세션 상태에 저장
                    st.session_state["one"] = one
                    st.session_state["draw_gdf_5179"] = draw_gdf_5179
                    st.session_state["ranking_results_4326"] = ranking_results_4326
                    st.session_state["matched_route_5179"] = matched_route_5179
                    st.session_state["matched_route_4326"] = matched_route_4326
                    st.session_state["num_ranks"] = num_ranks

                    st.session_state["generated_route"] = True
                    
                    st.balloons() # 매칭 성공 효과
                    # st.rerun() # 화면을 즉시 갱신하여 결과 지도를 하단에 로드합니다.
                    
            except Exception as e:
                import traceback
                st.error("❌ 작도 코스 매칭 중 시스템 내부 오류가 발생했습니다.")
                st.code(traceback.format_exc(), language="python")
    else:
        st.caption("💡 지도 우측 상단의 꺾은선 아이콘을 클릭한 뒤, 지도 위에 마우스 클릭으로 선을 이어 나가고 마지막 점을 한 번 더 클릭하여 선을 완성해 주세요.")


# ==============================================================================
if st.session_state["generated_route"]:
    st.markdown("---")

    # 저장해둔 세션 변수들 복원
    one = st.session_state["one"]
    draw_gdf_5179 = st.session_state["draw_gdf_5179"]
    matched_route_5179 = st.session_state["matched_route_5179"]
    matched_route_4326 = st.session_state["matched_route_4326"]
    ranking_results_4326 = st.session_state["ranking_results_4326"]
    num_ranks = st.session_state["num_ranks"]
    cmap = plt.get_cmap('rainbow', num_ranks)

    single_mhd_error = modified_hausdorff_distance(
        matched_route_5179.geometry[0], 
        draw_gdf_5179.geometry[0], 
        num_samples=50
    )

    center = [ranking_results_4326.iloc[0].geometry.centroid.y, ranking_results_4326.iloc[0].geometry.centroid.x]
    m = folium.Map(location=center, zoom_start=15, titles = None)

    folium.TileLayer(
        tiles=tiles,
        overlay=True,
        control = False,
        attr="VWorld",
        name = "Vworld"
    ).add_to(m)

    # 1. 사용자가 처음 그렸던 선 (반투명 점선)
    layer = folium.GeoJson(
        one, # 이전에 생성해둔 위경도 버전의 one
        name=f'AI Drawing ({draw_gdf_5179.length[0]/1000:.2f}km)',
        zoom_on_click=True,
        style_function=lambda x: {'color': 'red', 'dashArray': '5, 5', 'weight': 3, 'opacity': 0.8},
        popup=folium.Popup(f"<b>AI Drawing</b><br>🏃‍♂️ 총 거리: {draw_gdf_5179.length[0]/1000:.2f}km", max_width=300)
    ).add_to(m)

    # 2. 알고리즘으로 보정된 실제 도로 경로 (실선)
    fg_prox = folium.FeatureGroup(name=f'최근접 코스 (MHD 오차: {single_mhd_error:.1f}m)',
                                show=True)
    folium.GeoJson(
        matched_route_4326,
        name='Matched Route',
        zoom_on_click=True,
        style_function=lambda x: {'color': '#0000FF', 'weight': 6, 'opacity': 0.8},
        popup=folium.Popup(f"<b>최근접 코스</b><br>🏃‍♂️ 총 거리: {matched_route_5179['Total_Length'][0]/1000:.2f}km", max_width=300)
    ).add_to(fg_prox)

    # 3. 출발점 마커
    folium.Marker(
        location=[matched_route_4326.geometry.apply(lambda g: Point(g.geoms[0].coords[0]))[0].y, matched_route_4326.geometry.apply(lambda g: Point(g.geoms[0].coords[0]))[0].x],
        popup=f'Start Point',
        icon=folium.Icon(color='green', icon='s', prefix='fa')
    ).add_to(fg_prox)

    # 4. 도착점 마커
    folium.Marker(
        location=[matched_route_4326.geometry.apply(lambda g: Point(g.geoms[-1].coords[-1]))[0].y, matched_route_4326.geometry.apply(lambda g: Point(g.geoms[-1].coords[-1]))[0].x],
        popup=f'End Point',
        icon=folium.Icon(color='red', icon='e', prefix='fa')
    ).add_to(fg_prox)

    fg_prox.add_to(m)

    # 5. 대안 경로 순위 목록 루프 렌더링
    for i, data in ranking_results_4326[:num_ranks].iterrows():
        if pd.isna(data['mhd_dist']):
            continue
        
        rank = i + 1
        # route_color = colors[i]
        route_color = get_color(i)
        
        # 💡 핵심: 1~5위 각각을 FeatureGroup으로 묶어 하나의 체크박스로 제어하게 만듭니다.
        fg = folium.FeatureGroup(name=f'🏆 {rank}위 경로 ({data['Total_Length']/1000:.2f}km)<br>(MHD 오차: {data["mhd_dist"]:.1f}m / 거리비: {(data['Total_Length']/1000)/(draw_gdf_5179.length[0]/1000):.2f})',
                                show=False)
        
        # 2. 알고리즘으로 보정된 실제 매칭 도로 (실선)
        folium.GeoJson(
            data['geometry'],
            zoom_on_click=True,
            style_function=lambda x, color=route_color: {'color': color, 'weight': 6, 'opacity': 0.8},
            popup=folium.Popup(f"<b>{rank}위 추천 코스</b><br>🏃‍♂️ 총 거리: {data['Total_Length']/1000:.2f}km", max_width=300)
        ).add_to(fg)

        # 3. 출발점 마커
        folium.Marker(
            location = [data['geometry'].geoms[0].coords[0][1], data['geometry'].geoms[0].coords[0][0]],
            popup=f'{rank}위 Start Point',
            icon=folium.Icon(color='green', icon='s', prefix='fa')
        ).add_to(fg)

        # 4. 도착점 마커
        folium.Marker(
            location=[data['geometry'].geoms[-1].coords[-1][1], data['geometry'].geoms[-1].coords[-1][0]],
            popup=f'{rank}위 End Point',
            icon=folium.Icon(color='red', icon='e', prefix='fa')
        ).add_to(fg)
        
        # 완성된 그룹을 지도에 추가
        fg.add_to(m)

    # 레이어 컨트롤 추가
    folium.LayerControl(collapsed=False).add_to(m)

    # 측정 자(Measure Control) 플러그인 추가
    m.add_child(MeasureControl(
        position='topleft',          # 위치 지정 ('topleft', 'topright', 'bottomleft', 'bottomright')
        primary_length_unit='meters', # 기본 거리 단위
        secondary_length_unit='miles',# 보조 거리 단위
        primary_area_unit='sqmeters', # 기본 면적 단위
        active_color='#ff0000',       # 측정 선 색상 (빨간색)
        completed_color='#00ff00'     # 측정 완료 후 선 색상 (초록색)
    ))

    # 지도가 보여질 최적의 연산 범위 핏팅
    bounds = layer.get_bounds()
    m.fit_bounds(bounds, padding=[50, 50])

    # 6. Streamlit 화면 렌더링
    with st.container():
        st.markdown(f"### 🗺️ 최적 매칭 분석 지도")
        st.caption("우측 상단의 레이어 제어판에서 순위별 코스를 켜고 끌 수 있으며, 좌측 상단의 자 툴로 거리를 측정할 수 있습니다.")
        st_folium(m, use_container_width=True, height=600, returned_objects=[])
