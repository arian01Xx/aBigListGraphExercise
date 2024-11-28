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

class Solution{
private:
    void dfs(int node, int parent, int& timer, vector<int>& dis, vector<int>& low,
             vector<vector<int>>& result, vector<int> adj[], vector<bool>& visited){
        visited[node]=true;
        disc[node]=low[node]=timer++;
        
        for(auto nbr: adj[node]){
            if(nbr==parent){
                continue;
                if(!visited[nbr]){
                    dfs(nbr,node,timer,disc,low,result,adj,visited);
                    low[node]=min(low[node],low[nbr]);
                    if(low[nrd]>dis[node]){
                        vector<int> ans;
                        ans.push_back(node);
                        ans.push_back(nbr);
                        result.push_back(ans);
                    }
                }else{
                    low[node]=min(low[node],disc[nbr]);
                }
            }
        }
    }
public:
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
        vector<int> adj[n];
        for(int i=0; i<connections.size(); i++){
            int u=connections[i][0];
            int v=connections[i][1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        int timer=0;
        vector<int> disc(n,-1);
        vector<int> low(n,-1);
        int parent=-1;
        vector<bool> visited(n);

        vector<vector<int>> result;
        for(int i=0; i<n; i++){
            if(!visited[i]){
                dfs(i,parent,timer,disc,low,result,adj,visited);
            }
        }
        return result;
    }
};

class SolutionOne{
private:
    bool dfs(int curr, vector<char>& color, vector<int> adjList[], vector<int>& ans){
        color[curr]='G';
        for(int next: adjList[curr]){
            if(color[next]=='G' || (color[next]=='W' && dfs(next,color,adjList,ans))){
                return true;
            }
        }
        ans.push_back(curr);
        color[curr]='B';
        return false;
    }
public:
    vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
        for(int i=0; i<n; i++){
            if(group[i]==-1) group[i]=m++; //reemplaza el -1 por la cantidad de columnas +1
        }

        vector<int> itemList[n], groupList[m], itemOrder, groupOrder, ans;
        for(int i=0; i<n; i++){
            for(int j: beforeItems[i]){
                itemList[j].push_back(i);
                if(group[i]!=group[j]){
                    groupList[group[j]].push_back(group[i]);
                }
            }
        }

        vector<char> itemVisited(n,'W'), groupVisited(m,'W');
        for(int i=0; i<n; i++){
            if(itemVisited[i]=='W' && dfs(i,itemVisited,itemList,itemOrder)){
                return {};
            }
        }
        for(int i=0; i<m; i++){
            if(groupVisited[i]=='W' && dfs(i,groupVisited,groupList,groupOrder)){
                return {};
            }
        }

        reverse(itemOrder.begin(), itemOrder.end());
        reverse(groupOrder.begin(), groupOrder.end());
        unordered_map<int,vector<int>> map;
        for(int it: itemOrder){
            map[group[it]].push_back(it);
        }
        for(int it: groupOrder){
            for(int j: map[it]) ans.push_back(j);
        }
        return ans;
    }
};

class SolutionTwo{
public:
    int maxCandies(vector<int>& status, vector<int>& candies, vector<vector<int>>& keys, vector<vector<int>>& containedBoxes, vector<int>& initialBoxes) {
        int n=status.size();
        int ans=0;
        vector<int> vis(n,0);
        queue<int> q;

        for(int i=0; i<initialBoxes.size(); i++){
            if(status[initialBoxes[i]]) q.push(initialBoxes[i]);
            else vis[initialBoxes[i]]=1;
        }

        while(!q.empty()){
            int top=q.front();
            q.pop();
            ans+=candies[top];

            for(auto it: containedBoxes[top]){
                if(status[it]){
                    q.push(it);
                    status[it]=0;
                }else vis[it]=1;
            }
            for(auto it: keys[top]){
                if(vis[it] && !status[it]){
                    q.push(it);
                }
                status[it]=1;
            }
        }
        return ans;
    }
};

#define pl pair<int,int>
#define pll pair<pair<int,int>,int>
#define plp pair<int,pair<int,int>>

class SolutionThree{
public:
    int minCost(vector<vector<int>>& grid) {
        /*1 the cell to the right | i,j | i,j+1
          2 to the left           | i,j | i,j-1
          3 to the lower          | i,j | i+1,j
          4 to the upper cell     | i,j | i-1,j
          start cell (0,0),  ends cell (m-1,n-1)*/
        int n=grid.size();
        int m=grid[0].size();

        map<pl,vector<pll>> graph; //pair<int,int> | pair<pair<int,int>,int>
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                if(j<m-1) graph[{i,j}].push_back({{i,j+1},grid[i][j]==1 ? 0:1}); //si no es ni 1 ni 2 ni 3 ni 4, será 1
                if(i<n-1) graph[{i,j}].push_back({{i+1,j},grid[i][j]==3 ? 0:1});
                if(j>0) graph[{i,j}].push_back({{i,j-1},grid[i][j]==2 ? 0:1});
                if(i>0) graph[{i,j}].push_back({{i-1,j},grid[i][j]==4 ? 0:1});
            }
        }
        priority_queue<plp,vector<plp>,greater<plp>> q;  //pair<int,pair<int,int>> | distancia, coordenadas x y
        q.push({0,{0,0}});
        vector<vector<int>> dist(n,vector<int>(m,1e9));
        dist[0][0]=0;

        while(!q.empty()){
            auto node=q.top().second;
            int distance=q.top().first;
            q.pop();
            if(distance!=dist[node.first][node.second]) continue; //se ignora así mismo
            for(auto& it: graph[node]){ //coordenadas | proximas coordenadas y 0 o 1
                auto adj=it.first;
                int weight=it.second;
                int x=adj.first; //coordenada x
                int y=adj.second; //coordenada y
                if(weight+distance < dist[x][y]){
                    dist[x][y]=weight+distance;
                    q.push({weight+distance,adj}); //el menor peso posible, coordenadas
                }
            }
        }
        return dist[n-1][m-1]==1e9 ? -1: dist[n-1][m-1];
    }
};

class SolutionFour{
public:
    double frogPosition(int n, vector<vector<int>>& edges, int t, int target) {
        vector<vector<int>> graph(n+1);
        vector<bool> vis(n+1,false);
        for(int i=0; i<edges.size(); i++){
            int u=edges[i][0];
            int v=edges[i][1];
            graph[u].push_back(v);
            graph[v].push_back(u);
        }

        graph[1].push_back(0);
        queue<pair<int,double>> q;
        q.push({1,1.00});
        vis[1]=true;
        int level=0;

        while(!q.empty()){
            int len=q.size();
            while(len--){
                pair<int,double> currentNode=q.front();
                q.pop();
                int currentNodeIdx=currentNode.first;
                double currentNodeProb=currentNode.second;
                if(currentNodeIdx==target){
                    if(graph[currentNodeIdx].size()==1 && t>=level){
                        return currentNodeProb;
                    }else if(level==t) return currentNodeProb;
                    else return 0.00;
                }
                int s=graph[currentNodeIdx].size()-1;
                for(int i=0; i<graph[currentNodeIdx].size(); i++){
                    int child=graph[currentNodeIdx][i];
                    double prob=(double)currentNodeProb/s;
                    if(!vis[child] && child>0){
                        q.push({child,prob});
                        vis[child]=true;
                    }
                }
            }
            level++;
        }
        return 0.00;
    }
};

class SolutionFive{
public:
    bool parseBoolExpr(string expression) {
        stack<char> st;
        for(char currChar: expression){
            if(currChar==',' || currChar=='(') continue;
            if(currChar=='t' || currChar=='f' || currChar=='!' || currChar=='&' ||
               currChar=='|') st.push(currChar);
            else if(currChar==')'){
                bool hasTrue=false, hasFalse=false;
                while(st.top()!='!' && st.top()!='&' && st.top()!='|'){
                    char topValue=st.top();
                    st.pop();
                    if(topValue=='t') hasTrue=true;
                    if(topValue=='f') hasFalse=true;
                }
                char op=st.top();
                st.pop();
                if(op=='!'){
                    st.push(hasTrue ? 'f':'t');
                }else if(op=='&'){
                    st.push(hasFalse ? 'f':'t');
                }else{
                    st.push(hasTrue ? 't':'f');
                }
            }
        }
        return st.top()=='t';
    }
};

//PSEUDO CODIGO DE ARRIBA
/*
Approach
Initialize stack for operators & boolean values
Traverse through expression:
    If char is comma, or an open parenthesis, skip
    If char is bool, or an operator, push to stack
    If char is a closing parenthesis:
        Initialize two boolean flags to track presence of true & false within the parentheses
        Process values in parentheses:
            While the top of stack is not an operator:
                Pop from stack and check:
                    If t: hasTrue
                    If f: hasFalse
            Pop operator from stack
            Evaluate subexpression based on the operator:
                If !, push f if hasTrue. Otherwise, push t.
                If &, push f if hasFalse. Otherwise, push t.
                If |, push t if hasTrue is true. Otherwise, push f.
The final result will be at the top of the stack
*/

//ALGORITMO KRUSKAL CON LA ESTRUCTURA DE UNION-FIND O DISJOINT SET UNION(DSU)
class SolutionSix{
private:
    struct Edge{
        //origen, destino, peso, indice original
        int src, dst, wt, serial;
        Edge(int s, int d, int w, int n){
            src=s;
            dst=d;
            wt=w;
            serial=n;
        }
    };

    vector<int> par;
    vector<int> rank;
    //PATH COMPRESSION
    void initialize(int n){
        par=vector<int>(n); //representates lideres de cada grupo
        rank=vector<int>(n);//posicion de los conjuntos
        for(int i=0; i<n; i++){
            par[i]=i;
            rank[i]=0;
        }
    }

    void unionn(int x, int y){
        int x_rep=find(x), y_rep=find(y);
        if(x_rep==y_rep) return;
        if(rank[x_rep] < rank[y_rep]) par[x_rep]=y_rep;
        else if(rank[x_rep] > rank[y_rep]) par[y_rep]=x_rep;
        else{
            par[y_rep]=x_rep;
            rank[x_rep]++;
        }
    }

    int find(int x){ //donde cada nodo es su propio representante inicialmente
        if(x==par[x]) return x;
        par[x]=find(par[x]);
        return par[x];
    }

    //DSU IMPLEMENTATION
    vector<Edge> edge;
    static bool myCmp(Edge& e1, Edge& e2){
        return e1.wt < e2.wt;
    }

    int findMSTWt(int n, int m, int include, int exclude){
        int ret=0;
        initialize(n);
        int i, s=0;
        if(include != -1){
            Edge e=edge[include];
            unionn(e.src, e.dst);
            ret+=e.wt;
            s++;
        }
        //FINDING MST USING KRUSKAL'S ALGORITHM
        for(int i=0; s<n-1 and i<m; i++){
            if(i==exclude) continue;
            Edge e=edge[i];
            int x=find(e.src);
            int y=find(e.dst);
            if(x!=y){
                ret+=e.wt;
                unionn(x,y);
                s++;
            }
        }
        if(s<n-1) ret = 1e8;
        return ret;
    }
public:
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        //USING KRUSKAL'S ALGORITHM
        int m=edges.size();
        for(int i=0; i<m; i++){
            //primer nodo, segundo nodo, peso, posicion 
            Edge e(edges[i][0], edges[i][1], edges[i][2], i);
            edge.push_back(e);
        }
        //ordenamiento bajo el criterio de myCmp
        sort(edge.begin(), edge.end(), myCmp);
        vector<vector<int>> ans(2);

        int mn=findMSTWt(n, m, -1, -1);
        for(int i=0; i<m; i++){
            int curr1=findMSTWt(n, m, -1, i);
            if(curr1>mn){
                ans[0].push_back(edge[i].serial);
                continue;
            }
            int curr2=findMSTWt(n, m, i, -1);
            if(curr2==mn) ans[1].push_back(edge[i].serial);
        }
        return ans;
    }
};

//UTILIZA BITMASK, KRUST'S ALGORITHM AND DINAMYC PROGRAMMING
class SolutionSeven{
private:
    unordered_map<int,vector<int>> adj; //alimentas el unordered_map desde la f(x) main
    int dp[1<<15];

    int solve(int mask, int n, int k){ //prrimer nodo, total de cursos, paralelos
        if(mask==(1<<n)-1) return 0;
        if(dp[mask]!=-1) return dp[mask];
        vector<int> indegree(n,0); //para cada nodo
        for(int i=0; i<n; i++){
            if(!(mask & (1<<i))){
                for(auto& child: adj[i]){
                    indegree[child]++;
                }
            }
        }

        int coursesToTake=0;
        for(int i=0; i<n; i++){
            if(indegree[i]==0 && (mask & (1<<i))==0){
                coursesToTake |= (1<<i);
            }
        }

        int coursesCnt=__builtin_popcount(coursesToTake); //cantidad de unos
        int ans=INT_MAX;

        if(coursesCnt>k){ //mayores a la cantidade cursos iguales
            for(int i=coursesToTake; i>=0; i--){
                int combination=(i & coursesToTake); 
                int cnt=__builtin_popcount(combination);
                if(cnt!=k) continue;
                ans=min(ans, 1+solve(mask|combination,n,k));
            }
            return dp[mask]=ans;
        }
        return dp[mask]=1+solve(mask|coursesToTake,n,k);
    }
public:
    int minNumberOfSemesters(int n, vector<vector<int>>& relations, int k) {
        for(auto& x: relations){
            adj[x[0]-1].push_back(x[1]-1);
        }
        memset(dp,-1,sizeof(dp));//array, llenado, calculo del tamaño bytes
        return solve(0,n,k);
    }
};

int main(){
	return 0;
}