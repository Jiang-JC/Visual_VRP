from flask import Flask, request, jsonify
from TSP.TSP_solve import tsp_solve
from VRP.vrp_solve import vrp_solve

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def handle_data():
    data = request.get_json(silent=True)  # 使用silent=True安静地返回None，如果解析出错
    if not data or 'x' not in data or 'y' not in data:
        print("Received data is invalid or missing 'value':", data)
        return jsonify({"error": "Bad request"}), 400  # 返回400 Bad Request错误
    
    input_data = [list(pair) for pair in zip(data["x"], data["y"])]
    input_demand = data["demands"]

    print(input_data)
    print(data["demands"])

    if data["modelType"] == "TSP":
        output_data = tsp_solve(input_data)
    elif data["modelType"] == "VRP":
        output_data = vrp_solve(input_data, input_demand)

    print(output_data)
    processed_data = {"response": output_data}
    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)