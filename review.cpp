#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <cmath>
#include <map>
#include <set>

using namespace std;

//CUANDO USAR CADA ALGORITMO
/*
busqueda en anchura BFS-breadth first search
para encontrar la ruta mas corta en un grafo no ponderado, sin pesos

busqueda en profundidad DFS-depth first search
explorar todos los caminos posibles desde un nodo inicial hasta un
nodo objetivo
para detectar ciclos

busqueda en profundidad con implementacion iterativa
cuando necesitas la profundidad de una busqueda en profundidad pero con las 
ventajas de la busqueda en anchura, como evitar el uso excesivo de memoria

dijsktra
para encontrar la ruta mas corta desde un nodo a todos los demas

prim algoritmo
encontrar un arbol de expansion minima
minimizar todas las aristas tienen pesos y quieres minimizar
el coste de conexion de todos los nodos

floyd warshall algoritmo
para encontrar la ruta mas corta entre todos los pares de nodos
en un grafo ponderado
en grafos donde los pesos de las aristas pueden ser negativos
*/

//dijsktra's algorithm
vector<int> dijkstra(int node,vector<vector<int>> graph, vector<vector<pair<int,int>>> adj){
	//construccion de la lista de adyacencia
	for(const auto& it: graph){
		int u=it[0];
		int v=it[1];
		int w=it[2];
		adj[u].emplace_back(w,v);
		adj[v].emplace_back(w,u); //grafo no dirigido
	}

	//vector para almacenar las distancias
	int n=adj.size();
	vector<int> dist(n,INT_MAX);
	dist[node]=0;

	priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq; //dist, node
	pq.emplace(0,node);

	while(!pq.empty()){
		int currentDist=pq.top().first; //dist
		int u=pq.top().second; //node
		pq.pop();

		//si la distancia actual es mayor que la registrada, omitir
		if(currentDist >  dist[u]) continue;

		//revisar todos los vecinos de u
		for(auto& [weight,v]: adj[u]){
			if(dist[u]+weight < dist[v]){
				dist[v]=dist[u]+weight;
				pq.emplace(dist[v],v);
			}
		}
	}
	return dist;
}

//prim's algorithm | minimar un arbol de expansion minima con nodos
void prim(int n, int startNode, vector<vector<pair<int,int>>>& graph){
	vector<bool> visited(n, false);
	priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq; //peso, nodo
	visited[startNode]=true;
	for(auto &edge: graph[startNode]){
		int weight=edge.first;
		int node=edge.second;
		pq.push({weight,node});
	}
	while(!pq.empty()){
		auto[weight,node]=pq.top();
		pq.pop();
		if(!visited[node]){
			visited[node]=true;
			for(auto &edge: graph[node]){
				int newWeight=edge.first;
				int adjacentNode=edge.second;
				if(!visited[adjacentNode]){
					pq.push({newWeight,adjacentNode});
				}
			}
		}
	}
}

vector<vector<pair<int,int>>> convertGraph(int n, vector<vector<int>>& graph){
	vector<vector<pair<int,int>>> adjList(n);
	for(const auto &edge: graph){
		int node1=edge[0];
		int node2=edge[1];
		int weight=edge[2];

		adjList[node1].push_back({weight,node2});
		adjList[node2].push_back({weight,node1});
	}
	return adjList;
}

//floyd's wharshall | los pesos pueden ser negativos
void FloydWarshall(int n, vector<vector<int>>& nodes){
	vector<vector<int>> map(n, vector<int>(n,INT_MAX));

	//asignar el peso de las conexiones directas
	for(const auto& edge: nodes){
		int u=edge[0], v=edge[1], weight=edge[2];
		map[u][v]=weight;
		map[v][u]=weight; //debido a que es bidireccional
	}
	//la distancia de un nodo a si mismo es 0
	for(int i=0; i<n; i++){
		map[i][i]=0;
	}
	//algoritmo de floyd-warshall
	for(int k=0; k<n; k++){
		for(int i=0; i<n; i++){
			for(int j=0; j<n; j++){
				if(map[i][k]!=INT_MAX && map[k][j]!=INT_MAX){
					map[i][j]=min(map[i][j], map[i][k]+map[k][j]);
				}
			}
		}
	}
}

//busqueda en anchura
 void BFS(int startNode, vector<vector<int>>& graph){
 	vector<bool> visited(graph.size(), false);
 	queue<int> q;
 	visited[startNode]=true;
 	q.push(startNode);
 	while(!q.empty()){
 		int curr=q.front();
 		q.pop();
 		cout<<curr<<endl;
 		for(int neigh: graph[curr]){
 			if(!visited[neigh]){
 				visited[neigh]=true;
 				q.push(neigh);
 			}
 		}
 	}
 }

//busqueda en profundidad
 void DFS(int node, vector<vector<int>>& graph, vector<bool>& visited){
 	visited[node]=true;
 	for(int neighbor: graph[node]){
 		if(!visited[neighbor]){
 			DFS(neighbor,graph,visited); //recursividad
 		}
 	}
 }

//busqueda en profundidad con implementacion iterativa (no recursiva)
 void DFS_iterativo(int startNode, vector<vector<int>>& graph){
 	vector<bool> visited(graph.size(), false);
 	stack<int> s;
 	visited[startNode]=true;
 	s.push(startNode);
 	while(!s.empty()){
 		int curr=s.top();
 		s.pop();
 		cout<<"currently: "<<curr<<endl;
 		for(int neighbor: graph[curr]){
 			if(!visited[neighbor]){
 				visited[neighbor]=true;
 				s.push(neighbor); //insertar vecina en la pila
 			}
 		}
 	}
 }

//ejercicio modelo por excelencia
class Solution{
public:
	bool validPath(int n, vector<vector<int>>& edges, int source, int destination){
		vector<vector<int>> graph(n, vector<int>());
		for(const auto& edge: edges){
			graph[edge[0]].push_back(edge[1]);
			graph[edge[1]].push_back(edge[0]);
		}
		vector<bool> visited(n, false);
		visited[source]=true;
		queue<int> q;
		q.push(source);
		while(!q.empty()){
			int temp=q.front();
			q.pop();
			if(temp==destination) return true;
			for(int connect: graph[temp]){
				if(!visited[connect]){
					visited[connect]=true;
					q.push(connect);
				}
			}
		}
		return false;
	}
};

class SolutionOne{
public:
	int findJudge(int n, vector<vector<int>>& trust){
		vector<int> trustCount(n+1,0);
		vector<bool> trustSomeone(n+1,false);
		for(const auto& it: trust){
			trustSomeone[it[0]]=true;
			trustCount[it[1]]++;
		}
		for(int i=0; i<n; i++){
			if(!trustSomeone[i] && trustCount[i]==n-1){
				return i;
			}
		}
		return -1;
	}
};

class SolutionTwo{
public:
	void dfs(vector<vector<int>> graph, vector<vector<int>> paths, vector<int> path, int start, int destination){
		path.push_back(start);
		if(start==destination){
			paths.push_back(path);
		}
		for(auto x: graph[start]){
			dfs(graph,paths,path,x,destination);
		}
	}
	vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph){
		vector<vector<int>> paths;
		vector<int> path;
		int nodes=graph.size()-1;
		if(nodes==0) return paths;
		dfs(graph, paths, path, 0, nodes);
		return paths;
	}
};

class SolutionThree{
public:
	vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges){
		set<int> start;
		set<int> finish;
		for(int i=0; i<edges.size(); i++){
			for(int j=0; j<edges[i].size(); j++){
				start.insert(edges[i][0]);
				finish.insert(edges[i][1]);
			}
		}
		vector<int> alone;
		for(auto it=start.begin(); it!=start.end(); it++){
			if(finish.find(*it)==finish.end()){
				alone.push_back(*it);
			}
		}
		return alone;
	}
};

class SolutionFour{
public:
	long long maximumImportance(int n, vector<vector<int>>& roads){
		vector<int> deg(n,0);
		for(auto& e: roads){
			deg[e[0]]++;
			deg[e[1]]++;
		}
		sort(deg.begin(),deg.end());
		int ans=0;
		for(int i=0; i<n; i++){
			ans+=(i+1LL)*deg[i];
		}
		return ans;
	}
};

class SolutionFive{
public:
	//algoritmo de floyd-warshall
	int n, distanceThreshold;
	int dist[100][100];

	void FW(vector<vector<int>>& edges){
		fill(&dist[0][0], &dist[0][0]+100*100, 1e9);
		for(int i=0; i<n; i++){
			dist[i][i]=0;
		}
		for(auto& e: edges){
			int u=e[0], v=e[1], w=e[2];
			if(w<=distanceThreshold){
				dist[u][v]=dist[v][u]=w;
			}
		}
		for(int k=0; k<n; k++){
			for(int i=0; i<n; i++){
				for(int j=0; j<n; j++){
					dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
				}
			}
		}
	}
	int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold){
		this->n=n;
		this->distanceThreshold=distanceThreshold;
		FW(edges); //actualizó el vector con las menores distancias
		int min_cnt=n, city=-1;
		for(int i=0; i<n; i++){
			int cnt=-1;
			for(int j=0; j<n; j++){
				if(dist[i][j]<=distanceThreshold){
					cnt++;
				}
			}
			if(cnt<=min_cnt){
				min_cnt=cnt;
				city=i;
			}
		}
		return city;
	}
};

class SolutionSix{
public:
	int minCostConnectPoints(vector<vector<int>>& points){
		vector<vector<pair<int,int>>> adj_list(points.size(), vector<pair<int,int>>());
		for(int i=0; i<points.size(); i++){
			vector<int> curr=points[i];
			for(int j=0; j<points.size(); j++){
				if(j==i) continue;
				adj_list[i].push_back({abs(curr[0]-points[j][0])+abs(curr[1]-points[j][1]), j});
			}
		}
		priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
		vector<int> visited(points.size(),0);
		pq.push({0,0});
		int ans=0;
		while(!pq.empty()){
			int curr=pq.top().second;
			int wt=pq.top().first;
			pq.pop();
			if(visited[curr]) continue;
			visited[curr]=true;
			ans+=wt;
			for(pair<int,int> neighb: adj_list[curr]){
				if(visited[neighb.second]) continue;
				pq.push(neighb);
			}
		}
		return ans;
	}
};

class SolutionSeven{
public:
	void DFS(int node, unordered_map<int,vector<int>> adj, unordered_map<int,bool>& visited,
		     int& edgesCount, int& verticesCount){
		visited[node]=true;
		verticesCount++;
		for(int adjNode: adj[node]){
			edgesCount++;
			if(!visited[adjNode]){
				DFS(adjNode,adj,visited,edgesCount,verticesCount);
			}
		}
	}
	int countCompleteComponents(int n, vector<vector<int>>& edges){
		unordered_map<int,vector<int>> adj;
		unordered_map<int,bool> visited;
		for(int i=0; i<edges.size(); i++){
			adj[edges[i][0]].push_back(edges[i][1]);
			adj[edges[i][1]].push_back(edges[i][0]);
		}
		int ans=0;
		for(int i=0; i<n; i++){
			int edgesCount=0;
			int verticesCount=0;
			if(!visited[i]){
				DFS(i,adj,visited,edgesCount,verticesCount);
				if(verticesCount*(verticesCount-1)==edgesCount) ans++;
			}
		}
		return ans;
	}
};

 class SolutionHeight{
 public:
 	int maximalNetworkRank(int n, vector<vector<int>>& roads){
 		vector<int> degree(n,0);
 		vector<vector<bool>> connected(n,vector<bool>(n,false));
 		for(auto& x: roads){
 			int i=x[0];
 			int j=x[1];
 			degree[i]++;
 			degree[j]++;
 			connected[i][j]=true;
 			connected[j][i]=true;
 		}
 		int ans=INT_MIN;
 		for(int i=0; i<n; i++){
 			for(int j=i+1; j<n; i++){
 				int i_rank=degree[i];
 				int j_rank=degree[j];
 				int total=i_rank+j_rank;
 				if(connected[i][j]){
 					total--;
 				}
 				ans=max(ans,total);
 			}
 		}
 		return ans;
 	}
 };

 class SolutionNine{
 public:
 	long long ans=0; int s;
 	int dfs(int i, int prev, vector<vector<int>>& graph, int people=1){
 		for(int& x: graph[i]){
 			if(x==prev) continue;
 			people+=dfs(x,i,graph);
 		}
 		if(i!=0) ans+=(people+s-1)/s;
 		return people;
 	}
 	long long minimumFuelCost(vector<vector<int>>& roads, int seats){
 		vector<vector<int>> graph(roads.size()+1); s=seats;
 		for(vector<int>& r: roads){
 			graph[r[0]].push_back(r[1]);
 			graph[r[1]].push_back(r[0]);
 		}
 		dfs(0,0,graph);
 		return ans;
 	}
 };

 class SolutionTen{
 public:
 	vector<int> eventualSafeNodes(vector<vector<int>>& graph){
 		int n=graph.size();
 		vector<int> adj[n];
 		vector<int> topo;
 		queue<int> q;
 		vector<int> inDegree(n,0);
 		for(int i=0; i<n; i++){
 			for(int j=0; j<graph[i].size(); j++){
 				adj[graph[i][j]].push_back(i);
 			}
 		}
 		for(int i=0; i<n; i++){
 			for(auto it: adj[i]){
 				inDegree[it]++;
 			}
 		}
 		for(int i=0; i<n; i++){
 			if(inDegree[i]==0) q.push(i);
 		}
 		while(!q.empty()){
 			int node=q.front();
 			q.pop();
 			topo.push_back(node);
 			for(auto it: adj[node]){
 				inDegree[it]--;
 				if(inDegree[it]==0) q.push(it);
 			}
 		}
 		sort(topo.begin(), topo.end());
 		return topo;
 	}
 };

 class SolutionEleven{
 public:
 	vector<int> parent;
 	vector<int> rank;
 	int find(int i){
 		if(i==parent[i]){
 			return i;
 		}
 		return parent[i]=find(parent[i]);
 	}
 	void Union(int x, int y){
 		int x_parent=find(x);
 		int y_parent=find(y);
 		if(x_parent==y_parent){
 			return;
 		}
 		if(rank[x_parent]>rank[y_parent]){
 			rank[y_parent]=x_parent;
 		}else if(rank[x_parent]<rank[y_parent]){
 			rank[x_parent]=y_parent;
 		}else{
 			rank[x_parent]++;
 			parent[y_parent]=x_parent;
 		}
 	}
 	vector<int> findRedudantConnection(vector<vector<int>>& edges){
 		parent.resize(edges.size()+1);
 		rank.resize(edges.size()+1,0);
 		for(int i=0; i<=edges.size(); i++){
 			parent[i]=i;
 		}
 		for(auto edge: edges){
 			int x_parent=find(edge[0]);
 			int y_parent=find(edge[1]);
 			if(x_parent==y_parent){
 				return edge;
 			}else{
 				Union(edge[0],edge[1]);
 			}
 		}
 	}
 };

 class SolutionTwelve{
 public:
 	void bfs(int node, unordered_map<int,vector<int>>& graph, unordered_set<int>& visited){
 		queue<int> q;
 		q.push(node);
 		visited.insert(node);
 		while(!q.empty()){
 			int curr=q.front();
 			q.pop();
 			for(int neighbor: graph[curr]){
 				if(visited.find(neighbor)==visited.end()){
 					visited.insert(neighbor);
 					q.push(neighbor);
 				}
 			}
 		}
 	}
 	int makeConnected(int n, vector<vector<int>>& connections){
 		if(connections.size() < n-1){
 			return -1; //no hay suficiente cable
 		}
 		//construccion del grafo
 		unordered_map<int,vector<int>> graph;
 		for(int i=0; i<connections.size(); i++){
 			graph[connections[i][0]].push_back(connections[i][1]);
 			graph[connections[i][1]].push_back(connections[i][0]);
 		}
 		//contar componentes conectados
 		unordered_set<int> visited;
 		int components=0;
 		for(int i=0; i<n; i++){
 			if(visited.find(i)==visited.end()){
 				bfs(i,graph,visited);
 				components++;
 			}
 		}
 		return components-1;
 	}
 };

 class SolutionThirteen{
 public:
 	vector<vector<int>> getAncestors(int n, vector<vector<int>>& edges){
 		vector<vector<int>> adj(n);
 		vector<vector<int>> ans(n);
 		for(auto &i: edges){
 			adj[i[0]].push_back(i[1]);
 		}
 		for(int par=0; par<n; par++){
 			queue<int> q;
 			q.push(par);
 			while(!q.empty()){
 				int f=q.front();
 				q.pop();
 				for(auto &child: adj[f]){
 					//es ancestro?
 					if(ans[child].size()==0 || ans[child].back()!=par){
 						ans[child].push_back(par);
 						q.push(child);
 					}
 				}
 			}
 		}
 		return ans;
 	}
 };

 class SolutionFourteen{
 public:
 	int dfs(int node, vector<vector<int>>& graph, vector<int>& quiet, 
 		    vector<int>& ans, vector<bool>& visited){
 		if(visited[node]) return ans[node];
 		
 		visited[node]=1;
 		ans[node]=node; //tal cual su posición, tal cual su nodo
 		for(auto& newnode: graph[node]){
 			int candidate=dfs(newnode,graph,quiet,ans,visited);
 			if(quiet[candidate]<quiet[ans[node]]){
 				ans[node]=candidate;
 			}
 		}
 		return ans[node];
 	}
 	vector<int> loudAndRich(vector<vector<int>>& richer, vector<int>& quiet){
 		int n=quiet.size();
 		vector<vector<int>> graph(n);
 		vector<int> ans(n,-1);
 		vector<bool> visited(n,0);
 		for(auto& edge: richer){
 			//en la posición del primero agregamos el mayor
 			//practicamente invierte el grafo
 			//lo que está en la fila 1 agrega el valor de la posición cero
 			graph[edge[1]].push_back(edge[0]); 
 		}
 		for(int i=0; i<n; i++){
 			if(!visited[i]){
 				dfs(i,graph,quiet,ans,visited);
 			}
 		}
 		return ans;
 	}
 };

 class SolutionFifteen{
 public:
 	int n;
 	void dfs(int i, vector<int>& vis, vector<vector<int>>& stones){
 		vis[i]=1;
 		for(int j=0; j<n; j++){
 			if(!vis[j]){
 				if(stones[i][0]==stones[j][0] || stones[i][1]==stones[i][1]){
 					dfs(j,vis,stones);
 				}
 			}
 		}
 	}
 	int removeStones(vector<vector<int>>& stones){
 		n=stones.size();
 		vector<int> vis(n,0);
 		int ans=0;
 		for(int i=0; i<n; i++){
 			if(!vis[i]){
 				dfs(i,vis,stones);
 				ans++;
 			}
 		}
 		return ans;
 	}
 };

 class SolutionSixteen{
 public:
 	int reachableNodes(int n, vector<vector<int>>& edges, vector<int>& restricted){
 		unordered_map<int,vector<int>> graph;
 		unordered_set<int> restrictedSet(restricted.begin(), restricted.end());

 		vector<bool> visited(n, false);

 		for(const auto& edge: edges){
 			graph[edge[0]].push_back(edge[1]);
 			graph[edge[1]].push_back(edge[0]);
 		}
 		int reachableCount=0;
 		queue<int> q;
 		q.push(0);
 		visited[0]=true;
 		while(!q.empty()){
 			int node=q.front();
 			q.pop();
 			reachableCount++;
 			for(int neighbor: graph[node]){
 				if(!visited[neighbor] && restrictedSet.find(neighbor)==restrictedSet.end()){
 					visited[neighbor]=true;
 					q.push(neighbor);
 				}
 			}
 		}
 		return reachableCount;
 	}
 /*
 Input: n = 7, edges = [[0,1],[1,2],[3,1],[4,0],[0,5],[5,6]], restricted = [4,5]
Output: 4
 */
 };

 class SolutionSeventeen{
 public:
 	int findChampion(int n, vector<vector<int>>& edges){
 		vector<int> deg(n,0);
 		for(auto& e: edges){
 			int w=e[1];
 			deg[w]++;  //la frecuencia del segundo elemento de cada vector
 		}
 		//se recorren todos los nodos solo con este bucle
 		vector<int> deg0;
 		for(int i=0; i<n; i++){
 			if(deg[i]==0){
 				deg0.push_back(i);
 			}
 		}
 		if(deg0.size()!=1){
 			return -1;
 		}else{
 			return deg0[0];
 		}
 	}
 };

 //A PROBLEM TOTAL DIFFERENT
 //objetivo: hacer una copia profunda no solo de los grafos sino tambien
 //de sus conexiones
 class Node{
 public:
 	int val;
 	vector<Node*> neighbors;
 	Node(){
 		val=0;
 		neighbors=vector<Node*>();
 	}
 	Node(int _val){
 		val=_val;
 		neighbors=vector<Node*>();
 	}
 	Node(int _val, vector<Node*> _neighbors){
 		val=_val;
 		neighbors=_neighbors;
 	}
 };

 class SolutionEighteen{
 public:
 	unordered_map<Node*,Node*> mp;
 	Node* cloneGraph(Node* node){
 		if(node==NULL) return NULL;
 		if(mp.find(node)==mp.end()){
 			mp[node]=new Node(node->val);
 			for(auto adj: node->neighbors){ //itera sobre el vector 1D
 				mp[node]->neighbors.push_back(cloneGraph(adj));
 			}
 		}
 		return mp[node];
 	}
 };

//APLICACION DE FLOYD WARSHALL
 class SolutionNineteen{
 public:
 	int D[26][26];
 	inline void FW(vector<char>& original, vector<char>& changed, vector<int>& cost){

 		fill(&D[0][0], &D[0][0]+26*26, INT_MAX); //llenado del array de 26*26
 		const int sz=original.size();

 		for(int i=0; i<sz; i++){
 			int row=original[i]-'a';
 			int col=changed[i]-'a';
 			D[row][col]=min(D[row][col], cost[i]);
 		}

 		for(int i=0; i<26; i++) D[i][i]=0;
 		for(int k=0; k<26; k++){
 			for(int i=0; i<26; i++){
 				for(int j=0; j<26; j++){
 					D[i][j]=min((long long)D[i][j], (long long)D[i][k]+D[k][j]);
 				}
 			}
 		}
 	}
 	long long minimumCost(string source, string target, vector<char>& original, 
 		                  vector<char>& changed, vector<int>& cost){
 		FW(original,changed,cost);
 		const int n=source.size();
 		long long ans=0;
 		for(int i=0; i<n; i++){
 			int row=source[i]-'a';
 			int col=target[i]-'a';
 			if(D[row][col]==INT_MAX) return -1;
 			ans+=D[row][col];
 		}
 		return ans;
 	}
 };

 class Solutiontwenty{
 public:
 	int minScore(int n, vector<vector<int>> roads){
 		vector<vector<pair<int,int>>> adj(n+1);
 		//crear lista de adyacencia
 		for(const auto& road: roads){
 			int u=road[0], v=road[1], dist=road[2]; //una forma mas simplificada
 			adj[u].emplace_back(v, dist);
 			adj[v].emplace_back(u, dist);
 		}
 		//inicializar el resultado con el valor grande
 		int result=INT_MAX; //INT_MAX=2147483647 en un sistema de 32 bits

 		//utilizamos BFS o DFS para explorar los caminos
 		vector<bool> visited(n+1, false);
 		queue<int> q;
 		q.push(1);
 		visited[1]=true;

 		while(!q.empty()){
 			int city=q.front();
 			q.pop();
 			//explorar todas las rutas desde la ciudad actual
 			for(const auto& [neighbor, dist]: adj[city]){
 				result=min(result, dist);
 				if(!visited[neighbor]){
 					visited[neighbor]=true;
 					q.push(neighbor);
 				}
 			}
 		}
 		return result;
 	}
 /*
 Input: n = 4, roads = [[1,2,9],[2,3,6],[2,4,5],[1,4,7]]
Output: 5
 */
 };

 class SolutionTwentyOne{
 public:
 	int fn(int i, int j, int x, int y){
 		int ans=j-i; //calcula la distancia
 		ans=min(j-i, abs( abs(x-i)+1+abs(y-j)));
 		if(x>=i && j>=y && x!=y){
 			ans=x-i+1+j-y;
 		}
 		if(ans==0) ans++;
 		return ans;
 	}
 	vector<int> countOfPairs(int n, int x, int y){
 		vector<int> ans(n);
 		int p=min(x,y);
 		y=max(x,y);
 		x=p;
 		map<int,int> m;
 		for(int i=1; i<=n; i++){
 			for(int j=i+1; j<=n; j++){
 				m[fn(i,j,x,y)]++;
 			}
 		}
 		int j=0;
 		for(auto &i: m){
 			ans[j++]=i.second*2;
 		}
 		return ans;
 	}
/*
Input: n = 5, x = 2, y = 4
Output: [10,8,2,0,0]
*/
 };

//a beautiful code
 //24/08/2024
 class SolutionTwentyTwo{
 public:
 	bool isBipartite(vector<vector<int>>& graph){
 		int n=graph.size();
 		vector<int> color(n,-1);
 		for(int i=0; i<n; i++){
 			if(color[i]==-1){
 				queue<int> q;
 				q.push(i);
 				color[i]=0;
 				while(!q.empty()){
 					int node=q.front();
 					q.pop();
 					for(int neighbor: graph[node]){
 						if(color[neighbor]==-1){
 							color[neighbor]=1-color[node];
 							q.push(neighbor);
 						}else if(color[neighbor]==color[node]){
 							return false;
 						}
 					}
 				}
 			}
 		}
 		return true;
 	}
 };

 class SolutionTwentyThree{
 public:
 	double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end){
 		vector<vector<pair<int,int>>> graph(n);
 		for(int i=0; i<edges.size(); i++){
 			int a=edges[i][0];
 			int b=edges[i][1];
 			double w= -log(succProb[i]);
 			graph[a].push_back({b,w});
 			graph[b].push_back({a,w});
 		}
 		//el vector de pesos arrancando desde la cifra mas alta
 		vector<double> dist(n, numeric_limits<double>::max());
 		dist[start]=0;
 		priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> heap;
 		heap.push({0,start});
 		while(!heap.empty()){
 			auto[d,u]=heap.top();
 			heap.pop();
 			if(d>dist[u]){
 				continue;
 			}
 			for(auto[v,w]: graph[u]){
 				if(dist[v]>dist[u]+w){
 					dist[v]=dist[u]+w;
 					heap.push({dist[v],v});
 				}
 			}
 		}
 		return dist[end]==numeric_limits<double>::max() ? 0:exp(-dist[end]);
 	}
 };
 
 class SolutionTwentyFour{
 public:
 	int networkDelayTime(vector<vector<int>>& times, int n, int k){
 		//crear lista de adyacencia
 		unordered_map<int, vector<pair<int,int>>> adjacency;
 		for(const auto& time: times){
 			int src=time[0]; 
 			int dst=time[1];
 			int t=time[2];
 			adjacency[src].emplace_back(dst,t);
 		}
 		//priority queue for dijkstra's algorithm
 		priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
 		pq.emplace(0,k);
 		set<int> visited;
 		int delays=0;
 		while(!pq.empty()){
 			auto[time,node]=pq.top();
 			pq.pop();
 			if(visited.count(node)) continue;
 			visited.insert(node);
 			delays=max(delays,time);
 			for(const auto& neighbor: adjacency[node]){
 				int neighborNode=neighbor.first;
 				int neighborTime=neighbor.second;
 				if(!visited.count(neighborNode)){
 					//aqui se va acumulando el tiempo
 					pq.emplace(time+neighborTime, neighborNode); 
 				}
 			}
 		}
 		//delays actualizado obviamente
 		return visited.size() == n ? delays: -1;
 	}
 };
 
 class SolutionTwentyFive{
 public:
 	int networkBecomeIdle(vector<vector<int>>& edges, vector<int>& patience){
 		int n=patience.size();
 		vector<vector<int>> graph(n);
 		vector<int> time(n,-1);
 		for(auto x: edges){
 			graph[x[0]].push_back(x[1]);
 			graph[x[1]].push_back(x[0]);
 		}
 		queue<int> q;
 		q.push(0);
 		time[0]=0;
 		while(q.size()){
 			int node=q.front();
 			q.pop();
 			for(auto child: graph[node]){
 				if(time[child]==-1){
 					time[child]=time[node]+1; 
 					//de esa forma se registra el tiempo transcurrido
 					q.push(child);
 				}
 			}
 		}
 		int res=0;
 		//time[i]*2 es ida y vuelta
 		for(int i=1; i<n; i++){
 			//se obtiene el numero de veces que el mensaje fue reenviado
 			int extraPayload=(time[i]*2-1)/patience[i];
 			//el ultimo momento en que el servidor i envió su mensaje antes de recibir respuesta
 			int lastOut=extraPayload*patience[i];
 			//tiempo total en el que el servidor i finalmente recibe la respuesta
 			//del mensaje que envió
 			int lastInt=lastOut+time[i]*2;
 			res=max(res,lastInt);
 		}
 		return res+1;
 	}
 };

//el beautiful code era bipartito
 //este es cuatripartito...
 class SolutionTwentySix{
 public:
 	vector<int> gardenNoAdj(int n, vector<vector<int>>& paths){
 		vector<vector<int>> graph(n, vector<int>());
 		for(const auto& it: paths){
 			graph[it[0]-1].push_back(it[1]-1);
 			graph[it[1]-1].push_back(it[0]-1);
 		}
 		//solo cuatro tipo de flores
 		vector<int> flowers(n,0);
 		for(int i=0; i<n; i++){
 			vector<bool> used(5,false); //solo 1 2 3 4
 			for(int neighbor: graph[i]){
 				used[flowers[neighbor]]=true;
 			}
 			for(int f=1; f<=4; f++){
 				if(!used[f]){
 					flowers[i]=f;
 					break;
 				}
 			}
 		}
 		return flowers;
 	}
 };

 class SolutionTwentySeven{
 public:
 	vector<int> findOrder(int n, vector<vector<int>>& prerequisites){
 		vector<int> courses;
 		if(n<=0) return {};

 		//1.initialize the graph
 		unordered_map<int,int> inDegree;
 		unordered_map<int,vector<int>> graph;
 		for(int i=0; i<n; i++){
 			inDegree[i]=0;
 			graph[i]={};
 		}

 		//2.build the graph
 		for(int i=0; i<prerequisites.size(); i++){
 			int parent=prerequisites[i][1], child=prerequisites[i][0];
 			graph[parent].push_back(child);
 			inDegree[child]=inDegree[child]+1;
 		}

 		//3.find all sources, cursos que no tienen prerequisitos
 		//o sea van adelante
 		queue<int> sources;
 		for(auto itr=inDegree.begin(); itr!=inDegree.end(); itr++){
 			if(itr->second==0){
 				sources.push(itr->first);
 			
 			}
 		}
 		
 		//4.add it to the courses and subtract one
 		while(!sources.empty()){
 			int vertex=sources.front();
 			sources.pop();
 			courses.push_back(vertex);
 			vector<int> children=graph[vertex];
 			for(int child: children){
 				inDegree[child]=inDegree[child]-1;
 				if(inDegree[child]==0){
 					sources.push(child);
 				}
 			}
 		}
 		if(courses.size()!=n){
 			courses.clear();
 		}
 		return courses;
 	}
 };

//similar to the beautiful code
 class SolutionTwentyEight{
 public:
 	bool possibleBipartition(int n, vector<vector<int>>& dislikes){
 		vector<vector<int>> graph(n+1, vector<int>());
 		for(const auto& it: dislikes){
 			graph[it[0]].push_back(it[1]);
 			graph[it[1]].push_back(it[0]);
 		}
 		vector<int> hate(n+1,-1);
 		for(int i=1; i<=n; i++){
 			if(hate[i]==-1){
 				queue<int> q;
 				q.push(i);
 				hate[i]=0;
 				while(!q.empty()){
 					int node=q.front();
 					q.pop();
 					for(int neighbor: graph[node]){
 						if(hate[neighbor]== -1){
 							hate[neighbor]=1-hate[node];
 							q.push(neighbor);
 						}else if(hate[neighbor]==hate[node]){
 							return false;
 						}
 					}
 				}
 			}
 		}
 		return true;
 	}
 };

class SolutionTwentyNine{
public:
	int find(vector<int>& parent, int a){
		if(a==parent[a]) return a;
		return parent[a]=find(parent,parent[a]);
		//solo devuelve si el indice es igual al entero
	}
	void Union(vector<int>& parent, int a, int b){
		int p1=find(parent,a);
		int p2=find(parent,b);
		parent[p1]=p2;
	}
	bool equationsPossible(vector<string>& equations){
		vector<int> parent(26);
		for(int i=0; i<26; i++){
			parent[i]=i;
		}
		//para desigualdad !=
		for(int i=0; i<equations.size(); i++){
			if(equations[i][1]=='!') continue;
			Union(parent,equations[i][0]-'a',equations[i][3]-'a');
		}
		//para igualdad ==
		for(int i=0; i<equations.size(); i++){
			if(equations[i][1]=='=') continue;
			if(find(parent,equations[i][0]-'a')==find(parent,equations[i][3]-'a')){
				return false;
			}
		}
		return true;
	}
};

class SolutionThirty{
public:
	vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies){
		map<string,int> indegree;
		map<string,vector<string>> adj;
		int n=ingredients.size();
		for(int i=0; i<n; i++){
			for(auto& s: ingredients[i]){
				adj[s].push_back(recipes[i]);
				indegree[recipes[i]]++; //solo cuenta cuantos ingredientes falta
				//para tal plato ñam ñam
			}
		}
		vector<string> ans;
		queue<string> q;
		for(auto& x: supplies) q.push(x);
		while(!q.empty()){
			string node=q.front();
			q.pop();
			//el suplemento accede al ingrendiente de adj
			for(auto& newnode: adj[node]){
				indegree[newnode]--;
				if(indegree[newnode]==0){
					q.push(newnode);
					ans.push_back(newnode);
				}
			}
		}
		return ans;
	}
};

class SolutionThirtyOne{
public:
	unordered_map<int,int> vis;
	vector<vector<int>> adj;
	unordered_map<int,unordered_map<int,int>> m;
	int n;
	void dfs(int s, int x){
		vis[s]++;
		for(auto& y: adj[s]){
			if(vis.find(y)==vis.end()){
				m[x][y]++;
				dfs(y,x);
			}
		}
	}
	vector<bool> checkIfPrerequisite(int n, vector<vector<int>>& a, vector<vector<int>>& q){
		this->n=n;
		this->adj.resize(n);
		vector<bool> ans;
		for(int i=0; i<a.size(); i++){
			adj[a[i][1]].push_back(a[i][0]);
		}
		for(int i=0; i<n; i++){
			dfs(i,i);
			vis.clear();
		}
		for(int j=0; j<q.size(); j++){
			if(m[q[j][1]].find(q[j][0]) != m[q[j][1]].end()){
				ans.push_back(true);
			}else{
				ans.push_back(false);
			}
		}
		return ans;
	}
};

class SolutionThirtyTwo{
public:
	bool DFS(int src, int time, unordered_map<int,int>& path, vector<bool>& visited, vector<vector<int>>& graph){
		path[src]=time; //registra el tiempo
		visited[src]=true;
		if(src==0) return true;
		for(auto adj: graph[src]){
			if(!visited[adj]){
				//el tiempo va incrementando en 1
				if(DFS(adj,time+1,path,visited,graph)){
					return true;
				}
			}
		}
		path.erase(src);
		return false;
	}
	int mostProfitablePath(vector<vector<int>>& edges, int bob, vector<int>& amount){
		int n=edges.size()+1;
		vector<vector<int>> graph(n);
		for(auto it: edges){
			graph[it[0]].push_back(it[1]);
			graph[it[1]].push_back(it[0]);
		}
		unordered_map<int,int> path; //registra el nodo al que a llegado con su tiempo
		vector<bool> visited(n,false);
		//la funcion se encargará de llenar el vector bool con cada nodo
		DFS(bob,0,path,visited,graph);
		queue<vector<int>> q; //push{src,time,income}
		q.push({0,0,0});
		visited.assign(n,false);
		int ans=INT_MIN;
		while(!q.empty()){
			int src=q.front()[0];
			int time=q.front()[1];
			int income=q.front()[2];
			q.pop();
			visited[src]=true;
			if(path.find(src)==path.end()){
				income+=amount[src];
			}else{
				if(time<path[src]){
					income+=amount[src];
				}else if(time==path[src]){
					income+=(amount[src]/2);
				}
			}
			if(graph[src].size()==1 && src!=0){
				ans=max(ans,income);
			}
			for(auto adj: graph[src]){
				if(!visited[adj]){
					q.push({adj,time+1,income});
				}
			}
		}
		return ans;
	}
};

class SolutionThirtyThree{
private:
	void dfs(vector<vector<int>>& adj, vector<bool>& vis, int curr, long long &count){
		if(vis[curr]) return;
		vis[curr]=true;
		count++;
		for(auto neighbor: adj[curr]){
			dfs(adj,vis,neighbor,count); //recursividad
		}
	}
public:
	long long countPairs(int n, vector<vector<int>>& edges){
		vector<vector<int>> adj(n);
		for(int i=0; i<edges.size(); i++){
			adj[edges[i][0]].push_back(edges[i][1]);
			adj[edges[i][1]].push_back(edges[i][0]);
		}
		vector<bool> vis(n,false);
		long long res=0;
		long long visCount=0;
		for(int i=0; i<n; i++){
			if(!vis[i]){
				long long count=0;
				dfs(adj,vis,i,count);
				res += count*(visCount-count);
				visCount -= count;
			}
		}
		return res;
	}
};

//a beautiful and creative code
class SolutionThirtyFour{
#define ll long long int
public:
	void dfs(vector<vector<int>>& graph, vector<bool>& visited, int& c, int& i){
		visited[i]=true;
		c++;
		for(int j=0; j<graph[i].size(); j++){
			if(!visited[graph[i][j]]) dfs(graph,visited,c,graph[i][j]);
		}
	}
	int maximumDetonation(vector<vector<int>>& bombs){
		int n=bombs.size();
		vector<vector<int>> graph(n);
		for(int i=0; i<n; i++){
			ll x1, y1, r1;
			x1=bombs[i][0]; y1=bombs[i][1]; r1=bombs[i][2];
			for(int j=0; j<n; j++){
				if(i!=j){ //todas las combinaciones posibles de pares diferentes
					ll x2, y2, r2;
					x2=abs(x1-bombs[i][0]);
					y2=abs(y1-bombs[i][1]);
					//solo los detonadores son agregados
					if(x2*x2+y2*y2<=r1*r1) graph[i].push_back(j);
				}
			}
		}
		int ans=INT_MIN;
		for(int i=0; i<n; i++){
			int c=0;
			vector<bool> visited(n,false);
			//simplemente cuenta los detonantes a escalada
			dfs(graph,visited,c,i);
			ans=max(ans,c);
		}
		return ans;
	}
};

class SolutionThirtyFive{
public:
	int edgeScore(vector<int>& edges){
		int n=edges.size();
		vector<long long> scores(n,0);
		for(int i=0; i<n; ++i){
			scores[edges[i]]+=i; //una formula muy simplificada y elegante
		}
		int ans=0;
		long long maxScore=0;
		for(int i=0; i<n; ++i){
			if(scores[i]>maxScore){
				maxScore=scores[i];
				ans=i;
			}
		}
		return ans;
	}
};

class SolutionThirtySix{
public:
	vector<string> watchedVideosByFriends(vector<vector<string>>& watchedVideos, vector<vector<int>>& friends, int id, int level){
		unordered_map<int,vector<string>> graphVideos;
		for(int i=0; i<watchedVideos.size(); i++){
			graphVideos[i]=watchedVideos[i];
		}
		unordered_map<int,vector<int>> graphConnected;
		for(int i=0; i<friends.size(); i++){
			graphConnected[i]=friends[i];
		}
		int n=friends.size();
		vector<string> ans;
		int c=0;
		vector<bool> visited(n,false);
		queue<int> q;
		visited[id]=true;
		q.push(id);
		while(!q.empty() && c<level){
			int size=q.size();
			while(size--){
				int node=q.front();
				q.pop();
				for(int neighbor: graphConnected[node]){
					if(!visited[neighbor]){
						visited[neighbor]=true;
						q.push(neighbor);
					}
				}
			}
			c++;
		}
		if(c==level){
			unordered_map<string,int> videoCount;
			while(!q.empty()){
				int node=q.front();
				q.pop();
				for(const string& video: graphVideos[node]){
					videoCount[video]++;
				}
			}
			vector<pair<string,int>> sortedVideos(videoCount.begin(), videoCount.end());
			sort(sortedVideos.begin(), sortedVideos.end(), [](const auto& a, const auto& b){
				if(a.second==b.second) return a.first < b.first;
				return a.second < b.second;
			});
			for(const auto& video: sortedVideos){
				ans.push_back(video.first);
			}
		}
		return ans;
	}
};

class SolutionThirtySeven{
public:
	bool canFinish(int numCourses, vector<vector<int>>& prerequisites){
		//el problema puede verse como un problema de deteccion de ciclos en un grafo dirigido
		//si hay un ciclo significa que es imposible tomar todos los cursos, ya que algun curso depende
		//de si mismo directa o indirectamente
		
		vector<vector<int>> graph(numCourses);
		vector<int> inDegree(numCourses,0);
		//construir el grafo dirigido
		for(const auto& it: prerequisites){
			//esto es obligatorio segun el problema
			graph[it[1]].push_back(it[0]);
			inDegree[it[0]]++;
		}
		queue<int> q;
		//agregar todos los nodos con grado de entrada 0 a la cola
		for(int i=0; i<numCourses; ++i){
			if(inDegree[i]==0) q.push(i);
		}
		int c=0; //contador de cursos que se pueden tomar
		while(!q.empty()){
			int node=q.front();
			q.pop();
			c++;
			for(int neighbor: graph[node]){
				inDegree[neighbor]--;
				if(inDegree[neighbor]==0) q.push(neighbor);
			}
		}
		return c==numCourses;
	}
};

class SolutionThirtyEight{
public:
	vector<int> shortestDistanceAfterQueries(int n, vector<vector<int>>& queries){
		unordered_map<int,vector<int>> graph(n);
		for(int i=0; i<n-1; i++){
			graph[i].push_back(i+1); //grafo de forma lineal unidireccional
		}
		vector<int> ans;
		for(auto& query: queries){
			int u=query[0]; 
			int v=query[1];
			graph[u].push_back(v); //actualizacion del grafo con los datos dados
			//BFS
			vector<bool> visited(n,false);
			queue<pair<int,int>> q;
			q.push({0,0});
			visited[0]=true;
			int shortesPath=n-1;
			while(!q.empty()){
				auto[node,dist]=q.front();
				q.pop();
				if(node==n-1){
					shortesPath=dist;
					break;
				}
				for(int neighbor: graph[node]){
					if(!visited[neighbor]){
						visited[neighbor]=true;
						q.push({neighbor,dist+1});
					}
				}
			}
			ans.push_back(shortesPath);
		}
		return ans;
	}
};

class SolutionThirtyNine{
public:
	vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& redEdges, vector<vector<int>>& blueEdges){
		vector<int> adjRed[n], adjBlue[n];
		for(auto it: redEdges){
			adjRed[it[0]].push_back(it[1]);
		}
		for(auto it: blueEdges){
			adjBlue[it[0]].push_back(it[1]);
		}
		set<pair<int,int>> vis;
		vis.insert({0,-1}); //marca el color cero como visitado
		vector<int> ans(n,-1);
		queue<pair<int,pair<int,int>>> q; //node length color
		q.push({0, {0,-1}});
		while(!q.empty()){
			int node=q.front().first;
			int length=q.front().second.first;
			int color=q.front().second.second;
			q.pop();
			if(ans[node]== -1){
				ans[node]=length;
			}
			//RED EDGE
			if(color!=0){
				for(auto it: adjRed[node]){
					if(vis.find({it,0}) == vis.end()){
						vis.insert({it,0});
						q.push({it, {length+1, 0}});
					}
				}
			}
			//BLUE EDGE
			if(color!=1){
				for(auto it: adjBlue[node]){
					if(vis.find({it,1}) == vis.end()){
						vis.insert({it,1});
						q.push({it, {length+1, 1}});
					}
				}
			}
		}
		return ans;
	}
};

class SolutionFourty{
public:
	int closesMeetingNode(vector<int>& edges, int node1, int node2){
		int n=size(edges);
		vector<int> m1(n,-1);
		vector<int> m2(n,-1);
		auto dfs=[&edges](int u, auto& memo)->void{
			int time=0;
			//o sea que aun no a sido visitado
			while(u != -1 && memo[u]==-1){
				memo[u] = time++;
				u=edges[u];
			}
		};
		dfs(node1, m1), dfs(node2, m2);
		int id=-1;
		int minMax=1e9;
		for(int i=0; i<n; i++){
			if(min(m1[i], m2[i]) > -1 && max(m1[i], m2[i])<minMax){
				minMax=max(m1[i], m2[i]);
				id=i;
			}
		}
		return id;
	}
};

class SolutionFourtyOne{
public:
	bool validateBinaryTreeNodes(int n, vector<int>& leftChild, vector<int>& rightChild){
		vector<int> inDegree(n,0);
		//contar las entradas de cada nodo
		for(int i=0; i<n; ++i){
			//en todo arbol binario solo debe haber una entrada por cada nodo
			if(leftChild[i] != -1){
				inDegree[leftChild[i]]++;
				if(inDegree[leftChild[i]] > 1) return false;
			}
			if(rightChild[i] != -1){
				inDegree[rightChild[i]]++;
				if(inDegree[rightChild[i]] > 1) return false;
			}
		}
		//verificar si hay un solo nodo raiz (nodo sin padre)
		int rootCount=0;
		int rootNode=-1;
		for(int i=0; i<n; ++i){
			if(inDegree[i]==0){
				rootCount++;
				rootNode=i; //este es el nodo padre
			}
		}
		//debe haber exactamente un nodo raíz
		if(rootCount != 1) return false;
		//verificar que no haya ciclos y que todos los nodos sean accesibles desde la raiz
		queue<int> q;
		vector<bool> visited(n,false);
		q.push(rootNode); //arranca desde el nodo padre encontrado
		visited[rootNode]=true;
		int count=0;
		while(!q.empty()){
			int node=q.front();
			q.pop();
			count++;
			if(leftChild[node] != -1){
				//si ya fue visitado entonces es un ciclo cerrado
				if(visited[leftChild[node]]) return false;
				visited[leftChild[node]]=true; //sino se le marca como visitado
				q.push(leftChild[node]);
			}
			if(rightChild[node] != -1){
				if(visited[rightChild[node]]) return false;
				visited[rightChild[node]]=true;
				q.push(rightChild[node]);
			}
		}
		//si todos los nodos fueron visitados, entonces es un arbol valido
		return count==n;
	}
};

class SolutionFourtyTwo{
public:
	vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges){
		if(n==1) return {0};
		vector<int> ans;
		queue<int> q;
		vector<vector<int>> g(n); //conector
		vector<int> cnt(n,0); //contador
		for(vector<int> edge: edges){  //adjacency list
			cnt[edge[0]]++;
			cnt[edge[1]]++;
			g[edge[0]].push_back(edge[1]);
			g[edge[1]].push_back(edge[0]);
		}
		for(int i=0; i<n; i++){
			if(cnt[i]==1) q.push(i); //todos los nodos solo deben tener una conexion?
		}
		while(n>2){  //n=1 and n=2 are base, hasta que quede uno o dos nodos
			int size=q.size();
			n=n-size;
			while(size--){
				int node=q.front();
				q.pop();
				for(int padosi: g[node]){
					cnt[padosi]--;
					if(cnt[padosi]==1) q.push(padosi);
				}
			}
			size--;
		}
		while(!q.empty()){
			int node=q.front();
			q.pop();
			ans.push_back(node);
		}
		return ans;
	}
};

class SolutionFourtyThree{
public:
	int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        //menor cantidad de paradas
        vector<vector<pair<int,int>>> adj(n);
        for(auto it: flights){
            adj[it[0]].push_back({it[1],it[2]}); //ajaja mas rapido que mi modo
        }
        vector<int> price(n, INT_MAX);
        price[src]=0;
        queue<pair<int,pair<int,int>>> q;
        q.push({0,{src,0}});
        while(!q.empty()){
            int stop=q.front().first;
            int node=q.front().second.first;
            int cost=q.front().second.second;
            q.pop();
            if(stop>k){
                continue;
            }
            for(auto it: adj[node]){
                if(stop<=k && cost+it.second<price[it.first]){
                    price[it.first]=cost+it.second;
                    q.push({stop+1, {it.first, cost+it.second}});
                }
            }
        }
        if(price[dst]==INT_MAX) return -1;
        return price[dst];
    }
};

class SolutionFourtyFour{
public:
	int maxStarSum(vector<int>& vals, vector<vector<int>>& edges, int k){
		int n=vals.size();
		//construcción del grafo
		vector<vector<int>> adj(n);
		for(const auto& edge: edges){
			//empareja el primer nodo start con su precio al nodo de llegada
			adj[edge[0]].push_back(vals[edge[1]]);
			adj[edge[1]].push_back(vals[edge[0]]);
		}

		int maxSum=INT_MIN;

		//explora cada nodo como centro de una estrella
		for(int i=0; i<n; ++i){
			vector<int> neighbors=adj[i]; //itera fila por fila
			sort(neighbors.begin(), neighbors.end(), greater<int>());
			int currentSum=vals[i]; //empieza con el valor del nodo central
			for(int j=0; j<min(k, (int)neighbors.size()); ++j){
				if(neighbors[j] > 0){
					currentSum+=neighbors[j];
				}else{
					break;
				}
			}
			maxSum=max(maxSum,currentSum);
		}
		return maxSum;
	}
};

class SolutionFourtyFive{

	//conbina dos coordenadas en un solo codigo
    long long make(long long x, long long y){
        return (x << 20) | y;
    }

    //actualiza las distancias y maneja la cola de prioridad
    void maybe(long long state, int dist,
                priority_queue<pair<int,long long>>& q,
                unordered_map<long long,int>& d){
        if(!d.count(state) || d[state]>dist){
            d[state]=dist;
            q.push({-dist,state});
        }
    }
public:
    int minimumCost(vector<int>& start, vector<int>& target, 
    	            vector<vector<int>>& specialRoads) {
        //start solo tiene dos elementos, tu posicion en un espacio 2d
        //target la posicion de mi objetivo en el mismo espacio
        //costo de mi posicion a alguna otra posicion:
        // |x2-x1| + |y2-y1|
        //el vector de vector {x1,y1,x2,y2,cost} indica una direccion especial
        //que va de x1,y1 hasta x2,y2 con su costo
        //retornar el costo minimo para ir de mi posicion a la posicion del target

        //mapa para almacenar conexiones especiales
        unordered_map<long long, vector<vector<int>>> con;

        //conjunto de todos los puntos relevantes
        unordered_set<long long> all={make(target[0], target[1])};

        for(const auto& v: specialRoads){
            con[make(v[0],v[1])].push_back({v[2], v[3], v[4]}); //el atajo
            all.insert(make(v[0],v[1])); //las primeras x coordenadas
        }

        //mapa para guardar distancias mínimas
        unordered_map<long long,int> d;

        //conjunto para marcar estados ya procesados
        unordered_set<long long> have;

        d[make(start[0], start[1])]=0;
        priority_queue<pair<int, long long>> q;
        q.push({0, make(start[0], start[1])});

        while(!q.empty()){
        	//extrae el estado con la menor distancia
            const long long state=q.top().second;
            const int x=state >> 20;
            const int y=state & 1048575; //1048575=2^20-1
            const int dist= -q.top().first;
            q.pop();

            //si ya proesamos este estado lo saltamos
            if(have.count(state)){
                continue;
            }
            have.insert(state);

            //si alcanzan el objetivo retorna la distancia
            if(x==target[0] && y==target[1]) return dist;

            //si hay carreteras especiales desde este estado
            if(con.count(state)){
                for(const auto& v: con[state]){
                    maybe(make(v[0], v[1]), dist+v[2],q,d);
                }
            }

            //considera moverse a todos los puntos relevantes directamente
            for(long long s: all){
                maybe(s, dist + abs((s >> 20) - x) + abs((s & 1048575) - y), q, d);
            }
        }
        return -1; //en caso de que no se encuentre camino
    }
};

class SolutionFourtySix{
public:
	int countRestrictedPaths(int n, vector<vector<int>>& edges){
		//edges: u,v,weight
		vector<pair<int,int>> adj[n+1];

		for(auto it: edges){
			//agregar en la posición del nodo inicial el nodo de llegada y el peso
			adj[it[0]].push_back({it[1],it[2]});
			adj[it[1]].push_back({it[0],it[2]});
		}

		//DIJKSTRA's algorithm ->
		vector<int> dis(n+1,INT_MAX);
		priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
		pq.push({0,n});
		dis[n]=0;
		while(!pq.empty()){
			auto [wt,node]=pq.top();
			pq.pop();
			for(auto [v, edgeWeight]: adj[node]){
				if(dis[node]+edgeWeight < dis[v]){ //condicional del problema
					dis[v]=dis[node] + edgeWeight;
					pq.push({dis[v], v});
				}
			}
		}

		//funcion DFS con programación dinamica
		//para contar el numero de caminos restringidos desde el nodo 1 hasta el nodo n
		vector<int> dp(n+1, -1);
		function<int(int)> dfs=[&](int node)->int{
			//casos base de recursión
			if(node==1) return 1;
			if(dp[node] != -1) return dp[node];
			int ways=0;
			for(auto [v,wt]: adj[node]){
				if(dis[node] < dis[v]) ways=(ways+dfs(v))%1000000007; //para evitar desbordamientos
			}
			return dp[node]=ways;
			//la programación dinamica utiliza memoria para evitar recalculos
			//almacenados en el vector dp
		};
		return dfs(n);
	}
};

class SolutionFourtySeven{
public:
	vector<int> minimunTime(int n, vector<vector<int>>& edges, vector<int>& disappear){
		//edges[i]=u,v,length
		//disappear el tiempo que un nodo desaparece para siempre
		vector<pair<int,int>> adj[n];
		for(auto it: edges){
			adj[it[0]].push_back({it[1], it[2]});
			adj[it[1]].push_back({it[0], it[2]}); //porque es un grafo no dirigido
		}
		vector<int> v(n,-1);
		unordered_set<int> s;
		priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> q;
		q.push({0,0});
		while(!q.empty()){
			int dis=q.top().first;
			int node=q.top().second;
			q.pop();
			if(s.find(node) != s.end()) continue;
			s.insert(node);
			v[node]=dis;
			for(auto it: adj[node]){
				if(s.find(it.first)==s.end() && disappear[it.first]>it.second+dis){
					q.push({it.second+dis, it.first});
				}
			}
		}
		return v;
	}
};

class SolutionFourtyEight{
public:
	int countPaths(int n, vector<vector<int>>& roads){
		vector<pair<int,int>> adj[n];
		for(auto& nbr: roads){
			adj[nbr[0]].push_back({nbr[1],nbr[2]});
			adj[nbr[1]].push_back({nbr[0],nbr[2]});
		}

		priority_queue<pair<long long,long long>, vector<pair<long long,long long>>, greater<pair<long long,long long>>> pq;
		vector<long long> dist(n, LLONG_MAX);
		vector<long long> ways(n,0);
		dist[0]=0;
		ways[0]=1;
		pq.push({0,0});
		const long long mod=1e9+7;

		while(!pq.empty()){
			long long dis=pq.top().first;
			long long node=pq.top().second;
			pq.pop();
			for(auto& it: adj[node]){
				long long adjNode=it.first;
				long long edw=it.second; //seria la distancia que nos da el problema
				if(dis+edw < dist[adjNode]){
					dist[adjNode]=dis+edw; //va reduciendo su rango
					pq.push({dis+edw,adjNode});
					ways[adjNode]=ways[node];
				}else if(dis+edw == dist[adjNode]){
					ways[adjNode]=(ways[adjNode]+ways[node])%mod;
				}
			}
		}
		return ways[n-1]%mod;
	}
};

class SolutionFourtyNine{
public:
    bool findSafeWalk(vector<vector<int>>& grid, int health) {
        int n=grid.size();
        int m=grid[0].size();

        priority_queue<pair<int,pair<int,int>>, vector<pair<int,pair<int,int>>>> qu;
        vector<vector<int>> vis(n, vector<int>(m,0));
        if(grid[0][0]==1){
            qu.push({health-1,{0,0}});
        }else{
            qu.push({health,{0,0}});
        }
        vis[0][0]=1;

        while(!qu.empty()){
            int val=qu.top().first;
            int row=qu.top().second.first;
            int col=qu.top().second.second;
            if(row==n-1 && col==m-1){
                if(val>=1) return true;
            }

            qu.pop();
            int drow[4]={0,-1,0,1};
            int dcol[4]={-1,0,1,0};

            for(int i=0; i<4; i++){
                int n_row=row+drow[i];
                int n_col=col+dcol[i];
                if(n_row>=0 && n_row<n && n_col>=0 && n_col<m && !vis[n_row][n_col] && grid[n_row][n_col]==0){
                    vis[n_row][n_col]=1;
                    qu.push({val,{n_row,n_col}});
                }else if(n_row>=0 && n_row<n && n_col>=0 && n_col<m && !vis[n_row][n_col] && grid[n_row][n_col]!=0){
                    vis[n_row][n_col]=1;
                    qu.push({val-1,{n_row,n_col}});
                }
            }
        }
        return false;
    }
};

class SolutionFifty{
private:
    int dfs(int x, int y, vector<vector<int>>& matrix, vector<vector<int>>& dp, vector<vector<bool>>& visited){
        if(dp[x][y]!=-1) return dp[x][y];
        int n=matrix.size(), m=matrix[0].size();

        visited[x][y]=true;
        int drow[4]={0,-1,0,1};
        int dcol[4]={-1,0,1,0};

        int cnt=1;
        for(int i=0; i<4; i++){
            int newX=x+drow[i];
            int newY=y+dcol[i];
            if(newX>=0 && newX<n && newY>=0 && newY<m && matrix[newX][newY]>matrix[x][y]){
                cnt=max(cnt, 1 + dfs(newX, newY, matrix, dp, visited));
            }
        }
        dp[x][y]=cnt;
        return cnt;
    }
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int n=matrix.size();
        if(n==0) return 0;
        int m=matrix[0].size();
        
        vector<vector<int>> dp(n, vector<int>(m,-1));
        vector<vector<bool>> visited(n,vector<bool>(m,false));

        int ans=0;
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                ans=max(ans, dfs(i,j,matrix,dp,visited));
            }
        }
        return ans;
    }
};

class SolutionFiftyOne{
public:
    string longestDiverseString(int a, int b, int c) {
        priority_queue<pair<int,char>> pq;
        if(a>0) pq.push({a,'a'});
        if(b>0) pq.push({b,'b'});
        if(c>0) pq.push({c,'c'});

        string result="";

        while(!pq.empty()){
            auto [count1, char1]=pq.top();
            pq.pop();

            if(result.size()>=2 && result.back()==char1 && result[result.size()-2]==char1){
                if(pq.empty()) break;
                auto [count2, char2]=pq.top();
                pq.pop();

                result+=char2;
                count2--;
                if(count2>0) pq.push({count2, char2});
                pq.push({count1, char1});
            }else{
                result+=char1;
                count1--; //el conteo se va actualizando
                if(count1>0) pq.push({count1, char1});
            }
        }
        return result;
    }
};

//DESDE AQUI HACIA ARRIBA
class SolutionFiftyTwo{
private:
    vector<int> id;
    vector<bool> visited;

    //LLENA EL VECTOR ID
    void personToSeat(vector<int>& row){
        int n=row.size();
        id.resize(n);
        for(int i=0; i<n; i++){
            id[row[i]]=i; //cada numero de row con su pareja, pero esta vez el nummero es el indice y la pareja el elemento
        }
    }

    int findPartner(int x){
        //si da uno es impar, si da 0 es par
        //si es impar retorna x-1, si es par retorna x+1
        return x & 1 ? x-1 : x+1;
    }

    int dfs(vector<int>& row, int idx, int partner){
        if(row[idx]==partner) return 0;
        
        int currPartner=findPartner(row[idx]); //si es par o impar
        int currPartnerIdx=id[currPartner]; //la pareja de row[idx]+-1;
        int currPartnerNext=findPartner(currPartnerIdx);

        visited[row[idx]]=visited[currPartner]=true; //los marca como visitados

        return 1+dfs(row,currPartnerNext, partner);
    }

    int swapCouples(vector<int>& row){
        int n=row.size();
        int swaps=0;
        visited.resize(n);

        for(int i=0; i<n; i+=2){
            if(!visited[row[i]]){ //basicamente nos vamos a la pareja de i
                visited[row[i]]=visited[findPartner(row[i])]=true;
                swaps+=dfs(row, i+1, findPartner(row[i]));
            }
        }
        return swaps;
    }
public:
    int minSwapsCouples(vector<int>& row) {
        int n=row.size(), ans=0;
        personToSeat(row);
        return swapCouples(row);
    }
};

class SolutionFiftyThree{
private:
    //AQUI LLENA EL VECTOR SUBTREE Y EL VECTOR HEIGHT
    void dfs(vector<int> adj[], int n, int node, int prev, vector<int>& height, vector<int>& subtree){
        subtree[node]++;
        for(auto& ele: adj[node]){
            if(ele!=prev){
                height[ele]=1+height[node];
                dfs(adj, n, ele, node, height, subtree);
                subtree[node]+=subtree[ele];
            }
        }
    }

    void rec(vector<int> adj[], int n, int node, int prev, vector<int>& subtree, vector<int>& dp){
        for(auto& ele: adj[node]){
            if(ele!=prev){
                dp[ele]=dp[node]-subtree[ele]+(n-subtree[ele]);
                rec(adj, n, ele, node, subtree, dp);
            }
        }
    }
public:
    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        vector<int> adj[n];
        for(auto& ele: edges){
            int u=ele[0], v=ele[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        vector<int> height(n,0), subtree(n,0);
        vector<int> dp(n,0);
        dfs(adj,n,0,-1,height,subtree);

        for(int i=0; i<n; i++){
            dp[0]+=height[i];
        }
        rec(adj,n,0,-1,subtree,dp);
        return dp;
    }
};

class SolutionFiftyFour{
public:
    int shortestPathLength(vector<vector<int>>& graph) {
        int v=graph.size();
        queue<pair<int,int>> q; //nodo, mascara bits
        
        for(int i=0; i<v; i++){
        	//1 desplazado i veces
            q.push({i, 1<<i}); //la mascara bits sirve para marcar i como visitado
        }
        int viss=(1<<v)-1; //todos los nodos han sido visitados

        vector<vector<bool>> vis(v, vector<bool>(viss+1, false));
        for(int i=0; i<v; i++){
            vis[i][1<<i]=true; //solo ellos mismos están visitados
        }
        int path=0;

        while(!q.empty()){
            int n=q.size();
            while(n--){
                auto temp=q.front();
                q.pop();
                int node=temp.first;
                int mask=temp.second; //el desplazado por bits
                if(mask==viss) return path; //si todos los nodos ya fueron visitados retornas la longitud del camino
                for(auto i: graph[node]){ 
                    int nxt=mask|(1<<i); //se actualiza la mascara para incluir el nodo 'i'
                    if(vis[i][nxt]) continue;
                    if(nxt==viss) return path+1;
                    q.push({i,nxt});
                    vis[i][nxt]=true;
                }
            }
            path++;
        }
        return -1;
    }
};

class SolutionFiftyFive{
public:
    int reachableNodes(vector<vector<int>>& edges, int maxMoves, int n) {
        vector<vector<pair<int,int>>> adj(n);
        for(auto i: edges){
            adj[i[0]].push_back({i[1],i[2]+1}); //el peso aumentado en uno
            adj[i[1]].push_back({i[0],i[2]+1});
        }
        vector<int> dist(n,1e8); 
        dist[0]=0;
        //{peso,nodo}
        priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> q;
        map<int,int> par;
        par[0]=0;
        q.push({0,0});

        while(!q.empty()){
            int node=q.top().second;
            int wt=q.top().first;
            q.pop();

            if(wt > maxMoves) continue;
            for(auto i: adj[node]){
                if(wt+i.second <= maxMoves && dist[i.first]>wt+i.second){
                    par[i.first]=node;
                    //se actualiza
                    dist[i.first]=wt+i.second; //igual a la suma de los dos pesos
                    q.push({dist[i.first], i.first});
                }
            }
        }
        int ans=0;
        for(auto i: edges){
            if(((par.count(i[0]) && par[i[0]]==i[1]) || (par.count(i[1]) && par[i[1]]==i[0]))){
                ans+=i[2]; //mas el peso
            }else if(dist[i[0]]==1e8 && dist[i[1]]!=1e8){
                ans+=maxMoves - dist[i[1]];
            }else if(dist[i[0]]!=1e8 && dist[i[1]]==1e8){
                ans+=maxMoves - dist[i[0]];
            }else if(dist[i[0]]!=1e8 && dist[i[1]]!=1e8){
                ans+=min(2*maxMoves - dist[i[0]]-dist[i[1]] , i[2]);
            }
        }
        for(auto i: dist){
            ans+=(i != 1e8);
        }
        return ans;
    }
};

class SolutionFiftySix{
private:
    vector<vector<int>> adj;
    set<pair<pair<int,int>,int>> st;
    int dp[51][51][2];

    int solve(int i, int j, int k,  int win){ //1,2,0,1 arranca con esos valores
        if(i==j) return 2; //el gato atrapa el raton
        else if(i==0) return 1; //el raton gana

        if(dp[i][j][k]!=-1) return dp[i][j][k];
        if(st.find({{i,j},k})!=st.end()) return 0;
        else st.insert({{i,j},k});

        bool won=false;
        bool draw=false;

        if(k==0){
            int temp;
            for(auto it: adj[i]){
                temp=solve(it,j,1,2); //el gato tiene que llegar al 1 para ganar y retorna 2
                if(temp==0) draw=true;
                else if(temp==1){
                    won=true;
                    break;
                }
            }
        }else{
            int temp;
            for(auto it: adj[j]){
                if(it==0) continue;
                temp=solve(i,it,0,1); //el raton tiene que llegar al 0 para ganar y retorna 1
                if(temp==0) draw=true;
                else if(temp==2){
                    won=true;
                    break;
                }
            }
        }
        st.erase({{i,j},k});
        if(won) return dp[i][j][k]=win;
        else if(draw) return dp[i][j][k]=0;
        else{
            if(win==1) return dp[i][j][k]=2;
            else return dp[i][j][k]=1;
        }
    }
public:
    int catMouseGame(vector<vector<int>>& graph) {
        //mouse node 1, cat node 2, glory hole node 0
        adj=graph;
        int ans;
        for(int turn=0; turn<50; turn++){
            for(int i=0; i<51; i++){
                for(int j=1; j<51; j++){
                    for(int k=0; k<2; k++){
                        if(dp[i][j][k]==0) dp[i][j][k]=-1;
                    }
                }
            }
            ans=solve(1,2,0,1);
        }
        return ans;
    }
};

void printSuccessful(){
	cout<<"Cleaning code! "<<endl;
	cout<<endl;
}

int main(){

	printSuccessful();

	return 0;
}

//DAG directed acyclic graph
/*
no tiene ciclos, es lineal, y va en una sola direccion no en ambos
*/

/*
OJALA LEAS ESTO ARIAN DEL FUTURO
1)IA red neuronal para acabar con las personas en el menor de los pasos posibles
solo imagina tener el cerebro de chatGPT y querer por ejemplo, acabar con 15 personas
a la vez... como lo harías
2)Las Ias deben tener una imaginacion creacional interna asi como nosotros creacionamos
imagenes, sonidos y demás de forma interna sin expresarlo de forma corporal.
supongo que esto ya me e aproximado en mi libro horror al cuerpo
*/

/*
no olvidar el programa del mercado de putas en boids
*/

/*
el conocimiento sobre el cerebro y mbti de mi cuaderno en un programa
a modo de quiz y juego para empezar a entrenar
*/